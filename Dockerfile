# syntax=docker/dockerfile:1.7

# Stage 1: build a static Linux binary with version metadata injected.
#
# Builder is pinned to golang:1.26-alpine to match the CI test/lint/vulncheck
# floor pinned in .github/workflows/ci.yml (`go-version: "1.26"`). The
# downstream-consumer floor in go.mod (`go 1.23.0`) is intentionally
# separate — see CHANGELOG entry for v1.0.3.
#
# Pinned by digest for reproducible builds (the tag is kept inline so
# Dependabot's docker ecosystem can resolve + bump both in lockstep).
# Digest is the multi-arch image index (covers amd64 + arm64).
FROM --platform=$BUILDPLATFORM golang:1.26-alpine@sha256:91eda9776261207ea25fd06b5b7fed8d397dd2c0a283e77f2ab6e91bfa71079d AS builder

ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
ARG COMMIT=none
ARG BUILD_TIME=unknown

WORKDIR /src

# Cache module downloads as a separate layer.
COPY go.mod go.sum* ./
RUN go mod download

COPY . .

# CGO_ENABLED=0 ensures a fully-static binary that runs on any glibc/musl
# variant. The runtime stage below is python:3.12-slim (Debian-based) so
# CGO would also work, but keeping the build static avoids surprises if
# we ever swap the base image again (e.g. distroless-python).
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build \
        -trimpath \
        -ldflags "-s -w \
            -X github.com/llm-init/llm-init/internal/version.Version=${VERSION} \
            -X github.com/llm-init/llm-init/internal/version.Commit=${COMMIT} \
            -X github.com/llm-init/llm-init/internal/version.BuildTime=${BUILD_TIME}" \
        -o /out/llm-init \
        ./cmd/llm-init

# Stage 2: runtime image. Pre-v1.1.0 we shipped on distroless static
# (~2 MB). v1.1.0 swaps to python:3.12-slim because HF downloads run
# through the `hf` CLI shipped by huggingface_hub (Python package);
# distroless-static has no Python interpreter and distroless-python's
# image surface is comparable, so we standardise on the upstream slim
# variant for predictable security updates.
#
# Pinned by digest (multi-arch image index covering amd64 + arm64); the
# tag stays inline so Dependabot can resolve + bump tag and digest together.
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS runtime

ARG VERSION=dev
ARG COMMIT=none
ARG BUILD_TIME=unknown

LABEL org.opencontainers.image.title="llm-init"
LABEL org.opencontainers.image.description="Unified model download and OpenAI-compat sidecar for LLM inference engines"
LABEL org.opencontainers.image.source="https://github.com/beclab/model-console"
LABEL org.opencontainers.image.url="https://github.com/beclab/model-console"
LABEL org.opencontainers.image.documentation="https://github.com/beclab/model-console/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${COMMIT}"
LABEL org.opencontainers.image.created="${BUILD_TIME}"

# Runtime essentials:
#   - ca-certificates: HTTPS to huggingface.co + cdn-lfs.huggingface.co
#   - tini:            PID 1 reaper so the `hf` subprocess gets SIGTERM
#                      cleanly when k8s shuts the container down (the Go
#                      binary forwards ctx.Done() but tini ensures
#                      orphaned children die with the pod)
#
# huggingface_hub >= 0.32 installs hf_xet as a core dependency and the
# Hub is now fully Xet-backed, so file transfers go through hf_xet by
# default. Earlier hf_xet releases bypassed tqdm (silent downloads), but
# current hf_xet renders the SAME unit_scale=B tqdm bar the LFS path uses
# (file_download.py xet_get), batched ~200ms. The reason the dashboard
# showed 0/0 was NOT xet itself: huggingface_hub builds the bar with
# tqdm(disable=is_tqdm_disabled()) which returns None on a normal run,
# and tqdm(disable=None) auto-suppresses on a non-TTY — and we capture
# the subprocess stderr via a pipe. internal/adapter/hfwrap.buildEnv
# sets TQDM_POSITION=-1 (huggingface_hub's force-enable hatch) so the
# byte bars are emitted to our pipe and parsed by the stderr scanner.
#
# v1.1.0 ships `hf` as the download driver (Go runs `hf download …`
# directly via internal/adapter/hfwrap); the Python wrapper script was
# deleted. The `hf` binary is exposed by huggingface_hub itself, so a
# bare `pip install huggingface_hub` is sufficient. Pinned to an exact
# 0.x release for reproducible builds: hfwrap parses the tqdm progress
# bars huggingface_hub emits, and the 1.0 line reworked that output, so
# the major is held at <1 until hfwrap's scanner is validated against it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir 'huggingface_hub==0.36.2' && \
    pip cache purge

# Copy the Go binary and pre-create the runtime data directories with
# UID 1000 ownership so freshly created Docker named volumes inherit
# the correct owner.
#
# v1.1.0 fixed container paths:
#   /cache/hf/hub                     -- HF Hub cache (rw); pinned via
#                                        HF_HUB_CACHE env below so the
#                                        path is UID-neutral and does
#                                        not depend on $HOME / /root.
#   /run/llm-init                     -- ephemeral sentinel + model_path,
#                                        plus the resolved model-spec.json
#                                        (MODEL_SPEC_PATH default). The spec
#                                        is seeded from MODEL_MODE (+ optional
#                                        MODEL_SUPPORTS) on first boot and is
#                                        editable via the dashboard.
#   /etc/llm-init                     -- optional mount point for a fuller
#                                        model-spec.json: bind/COPY a file
#                                        and point MODEL_SPEC_PATH at it to
#                                        let disk win over the env seed.
#
# IMPORTANT — HOME must point at a writable dir for hf_xet:
#   The Rust-implemented hf_xet wheel calls dirs::cache_dir() which
#   resolves to "$HOME/.cache". Kubernetes (unlike Docker) does NOT
#   honour /etc/passwd when picking HOME for runAsUser:1000 — it only
#   looks at the image's ENV. Without an explicit ENV HOME, hf_xet's
#   chunk staging hits "Permission denied (os error 13)" the first
#   time it tries to mkdir under "/.cache" or "/home/llminit/.cache".
#   We materialise /home/llminit, chown it, and pin HOME via ENV so
#   both Docker- and K8s-spawned containers behave identically.
#
# deploy/wrappers/common.sh rides along because llm-init publishes it into
# RUN_DIR at boot (handoff.PublishWrapperHelpers): the engine container
# sources that one copy instead of the copy its chart inlined, so a fix to
# supervise_engine ships with the image rather than with 40 charts. The
# file is COPYd rather than embedded so it stays the same file shellcheck
# and tests/wrappers already cover.
COPY --from=builder /out/llm-init /usr/local/bin/llm-init
COPY --from=builder /src/deploy/wrappers/common.sh /usr/share/llm-init/wrappers/common.sh
RUN mkdir -p /cache/hf/hub /run/llm-init /etc/llm-init /home/llminit && \
    groupadd -g 1000 llminit 2>/dev/null || true && \
    useradd  -u 1000 -g 1000 -d /home/llminit -s /usr/sbin/nologin llminit 2>/dev/null || true && \
    chown -R 1000:1000 /cache/hf /run/llm-init /etc/llm-init /home/llminit && \
    chmod 0755 /home/llminit /cache /cache/hf

# UID 1000 matches the conventional first-non-system UID and the
# Kubernetes PodSecurity / OpenShift "restricted-v2" expectations.
USER 1000:1000

# HOME is read by huggingface_hub (Python) and hf_xet (Rust) to derive
# the default cache location. Pinning it here means callers who don't
# inherit the Docker-style /etc/passwd lookup (Kubernetes pod runtime,
# nerdctl, some CI shells) still get a writable cache root.
ENV HOME=/home/llminit

# Pin the HF Hub cache to a fixed, UID-neutral path. v1.1.0 BREAKING:
# operators no longer override this -- the path is the deployment
# contract surfaced as a volume mount target.
# huggingface_hub's library reads HF_HUB_CACHE for the cache root; we
# bake the value here so the path stays consistent across all callers
# regardless of $HOME.
ENV HF_HUB_CACHE=/cache/hf/hub

# Default port matches the PORT env default.
EXPOSE 8080

# python:3.12-slim ships /bin/sh, but we keep the self-probe form so
# the healthcheck is identical across base-image rotations.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD ["/usr/local/bin/llm-init", "--healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/llm-init"]
