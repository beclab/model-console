# Makefile for llm-init.
#
# Targets:
#   make build               Build the local binary into bin/llm-init.
#   make test                Run unit tests with race detector.
#   make cover               Run tests and emit coverage.html.
#   make vet                 go vet ./... (matches the CI test job).
#   make fmt                 Apply gofmt + goimports via golangci-lint fmt.
#   make smoke               Run cmd/llm-init smoke tests only.
#   make test-download       Run the offline download suite (tag: download).
#   make test-wrappers       Run tests/wrappers (POSIX sh, fake engines).
#   make integration         Run integration tests (needs docker; matches CI).
#   make test-integration-download
#                            Local-only download-fault matrix (URL + ollama-url
#                            via faultsrv; needs docker; NOT in CI).
#   make test-integration-negative
#                            Local-only real-upstream negative cases (needs
#                            docker + network; NOT in CI).
#   make compat              Run Python client compat suite (needs docker + pip).
#   make fuzz                30s fuzz pass over parser corpora.
#   make bench               Run benchmarks for hot paths.
#   make lint                Run golangci-lint (matches CI; pin in tools/install-lint.sh).
#   make lint-fix            Run golangci-lint --fix (auto-fixes safe categories).
#   make compose-validate    docker compose config -q for every template.
#   make k8s-validate        kubeconform per engine (matches CI).
#   make vulncheck           govulncheck ./...
#   make docker              Build a single-arch Docker image.
#   make docker-multi-load   Build the image into the local docker engine
#                            (single platform, --load).
#   make docker-multi-push   Build and push a real multi-arch image.
#   make docker-multi        Print a notice — multi-arch needs --load or
#                            --push to produce anything usable.
#   make docker-run          Build and run the image with --version.
#   make clean               Remove build artifacts.
#   make tidy                Run go mod tidy.
#
# Override IMAGE / PLATFORMS for buildx:
#   make docker-multi-push PLATFORMS=linux/amd64,linux/arm64

SHELL          := /bin/sh
GO             ?= go
PKG            := github.com/llm-init/llm-init
BIN_DIR        := bin
BINARY         := $(BIN_DIR)/llm-init

# Version metadata injected at link time.
VERSION    ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
BUILD_TIME ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

LDFLAGS := -s -w \
    -X $(PKG)/internal/version.Version=$(VERSION) \
    -X $(PKG)/internal/version.Commit=$(COMMIT) \
    -X $(PKG)/internal/version.BuildTime=$(BUILD_TIME)

# Default the local `build` target to a statically linked binary so it
# matches what CI publishes (the release job in .github/workflows/ci.yml
# sets CGO_ENABLED=0 GOOS=linux GOARCH=$arch). A developer who copies
# bin/llm-init onto a distroless Pod gets the same artifact CI would
# have shipped, instead of a host-CGO-tainted build that links against
# /lib/libc.so.6 and fails to start.
#
# Override per-invocation if you actually need cgo (e.g. local race
# build): `make build CGO_ENABLED=1`.
#
# `make test` and `make cover` deliberately leave CGO_ENABLED unset so
# `-race` works (the race detector requires cgo). They sit above this
# block in the file but inherit the parent shell's environment, not
# this assignment, because we use `?=` on CGO_ENABLED so a caller's
# explicit value still wins.
CGO_ENABLED ?= 0

IMAGE     ?= docker.io/beclab/llm-init
IMAGE_TAG ?= $(VERSION)
PLATFORMS ?= linux/amd64

.PHONY: all build test cover vet fmt smoke verify test-download test-wrappers integration \
        test-integration-download test-integration-negative compat fuzz bench \
        lint lint-fix docker docker-run \
        docker-multi docker-multi-load docker-multi-push \
        compose-validate k8s-validate vulncheck clean tidy help

all: build

build: ## Build the local binary (CGO_ENABLED=0 by default; mirrors the CI release job)
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=$(CGO_ENABLED) $(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/llm-init

test: ## Run unit tests with race detector
	$(GO) test -race -count=1 ./...

verify: vet test test-download test-wrappers lint ## Every gate the CI docker job needs, before pushing a tag
	@# `docker` in .github/workflows/ci.yml needs test, lint, wrappers and
	@# the download suite among its `needs:`, so a tag whose CI fails any
	@# of them publishes no image. This target is that set.
	@#
	@# test-download is the one that has to be listed separately: it is
	@# behind `-tags download`, so `go test ./...` does not compile it, and
	@# an interface change that leaves its fakes behind is invisible to
	@# every other target here. v1.5.0 was tagged that way.
	@echo "verify: all gates passed"

cover: ## Run tests and write coverage.html
	$(GO) test -race -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "coverage report: coverage.html"

vet: ## go vet ./... (matches the CI test job)
	$(GO) vet ./...

fmt: ## Apply gofmt + goimports via `golangci-lint fmt` (matches .golangci.yml formatters)
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "golangci-lint not found; run: bash tools/install-lint.sh"; \
		exit 1; \
	}
	@# v2's `fmt` subcommand runs exactly the formatters configured under
	@# the `formatters:` block (gofmt + goimports with the local prefix),
	@# so dev formatting matches what the lint gate checks in CI.
	golangci-lint fmt

smoke: ## Run cmd/llm-init smoke tests only
	$(GO) test -count=1 -run=TestRun_ ./cmd/llm-init/

test-download: ## Run the offline download suite (tag: download; in-process mocks, no network/docker)
	@# Exercises every MODEL_SOURCE channel (url/hf/ollama/ollama-url)
	@# against the local faultsrv + mockollama + fakehf mocks. Hermetic:
	@# no external network, daemons, or docker. See tests/download/README.md.
	$(GO) test -tags download -timeout 5m ./tests/download/...

test-wrappers: ## Run the engine-wrapper tests (POSIX sh, fake engines, no docker)
	@# Same loop the CI wrappers job runs. Each test drives a wrapper or
	@# common.sh against fake engine binaries: no model, no GPU, no docker.
	@for t in tests/wrappers/*_test.sh; do \
		echo "== $$t"; \
		sh "$$t" || exit 1; \
	done

integration: ## Run integration tests (needs docker; llamacpp runs by default, see tests/integration/README.md)
	@# Mirrors the CI integration job. ollama/vllm/sglang suites self-skip
	@# unless their LLM_INIT_INTEGRATION_<ENGINE> env is set; llamacpp runs
	@# unless LLM_INIT_INTEGRATION_DISABLED is set. LLM_INIT_IMAGE selects
	@# the image under test (defaults to a freshly built local image in CI).
	$(GO) test -tags integration -timeout 12m -v ./tests/integration/...

test-integration-download: ## Local-only download-fault matrix (needs docker; URL + ollama-url via faultsrv, NOT in CI)
	@# Builds llm-init + faultsrv images on demand (override with
	@# LLM_INIT_IMAGE / FAULTSRV_IMAGE) and drives a per-case compose stack
	@# asserting GET /api/progress reaches the right phase + last_error for
	@# normal / retriable / non-retriable / sha-mismatch sources. Heavy and
	@# docker-bound, so it is gated out of CI; see tests/integration/README.md.
	LLM_INIT_INTEGRATION_DOWNLOAD=1 \
		$(GO) test -tags integration -timeout 30m -v \
		./tests/integration/download/... ./tests/integration/ollamaurl/...

test-integration-negative: ## Local-only real-upstream negative cases (needs docker + network, NOT in CI)
	@# Runs the production deploy/compose/<engine>.yml files against the live
	@# HuggingFace / Ollama upstreams: missing HF repo, bad HF revision,
	@# unknown Ollama tag. Asserts the operator-facing last_error. Gated out
	@# of CI because it hits real networks; see tests/integration/README.md.
	LLM_INIT_INTEGRATION_NEGATIVE=1 \
		$(GO) test -tags integration -timeout 20m -v \
		./tests/integration/negative/...

compat: ## Run Python client compat suite (needs docker + pip deps)
	python -m pip install -r tests/compat/requirements.txt
	pytest tests/compat -v

fuzz: ## 30s fuzz pass over the HF-repo parser corpus (FuzzParseHFRepo)
	@# -fuzz only accepts a single target per `go test` invocation. The 30s
	@# budget mirrors what is documented in CHANGELOG and is short enough for
	@# an opportunistic local pre-push run.
	$(GO) test -run=^$$ -fuzz=FuzzParseHFRepo -fuzztime=30s ./internal/config/

bench: ## Run benchmarks for hot paths (chat translate)
	$(GO) test -run=^$$ -bench=. -benchmem ./internal/adapter/ollama/

lint: ## Run golangci-lint (matches CI; non-zero exit on findings)
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "golangci-lint not found."; \
		echo "  Install (matches the CI pin):  bash tools/install-lint.sh"; \
		echo "  Or, if you use asdf / mise:    asdf install   # reads .tool-versions"; \
		echo ""; \
		echo "Both paths target the same version as .github/workflows/ci.yml;"; \
		echo "version skew between dev and CI is the bug they exist to close."; \
		exit 1; \
	}
	golangci-lint run --timeout=5m ./...

lint-fix: ## Run golangci-lint with --fix (auto-fix gofmt / goimports / misspell / unconvert)
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "golangci-lint not found; run: bash tools/install-lint.sh"; \
		exit 1; \
	}
	@# --fix only touches issues from linters that opt in; on this
	@# tree that means gofmt / goimports / misspell / unconvert.
	@# Everything else (errcheck, revive, govet, ...) still requires
	@# a human edit.
	golangci-lint run --fix --timeout=5m ./...

docker: ## Build single-arch image
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):$(IMAGE_TAG) \
		.

docker-run: docker ## Build then run --version
	docker run --rm $(IMAGE):$(IMAGE_TAG) --version

docker-multi: ## Print a notice — `buildx build` without --load/--push produces nothing usable
	@echo "make docker-multi: pick a sub-target."
	@echo "  make docker-multi-load   single-arch, ends up in your local docker engine."
	@echo "  make docker-multi-push   real multi-arch, requires a registry tag."
	@echo "Pre-v1.0.4 this target ran 'docker buildx build' with neither flag,"
	@echo "which left the build manifest in the buildx cache and produced no"
	@echo "image you could docker run — the same compose / k8s manifests then"
	@echo "fell back to docker.io/beclab/llm-init:latest at deploy time."
	@exit 0

docker-multi-load: ## Build single-arch image directly into the local docker engine
	@# `--load` only works with a single platform. We override PLATFORMS to
	@# the current host arch so a developer can `make docker-multi-load &&
	@# docker run llm-init:dev` without a registry round-trip.
	docker buildx build \
		--platform linux/$$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/') \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):$(IMAGE_TAG) \
		--load \
		.

docker-multi-push: ## Build true multi-arch image (PLATFORMS=$(PLATFORMS)) and push to $(IMAGE)
	@# `--push` requires the target tag to be a registry path; CI uses
	@# this with $(IMAGE)=docker.io/beclab/llm-init.
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):$(IMAGE_TAG) \
		--push \
		.

compose-validate: ## Validate every deploy/compose/*.yml against .env.example
	@for f in deploy/compose/*.yml; do \
		case "$$f" in *.gpu.yml|*.integration.yml) continue ;; esac; \
		echo "==> $$f"; \
		MODEL_NAME=ci-validate MODEL_SOURCE=hf://ci/repo \
			docker compose -f $$f --env-file deploy/compose/.env.example config -q || exit 1; \
	done
	@echo "==> deploy/compose/embed.yml + embed.gpu.yml"
	@MODEL_NAME=ci-validate MODEL_SOURCE=hf://ci/repo \
		docker compose -f deploy/compose/embed.yml -f deploy/compose/embed.gpu.yml \
			--env-file deploy/compose/.env.example config -q
	@echo "==> deploy/compose/rerank.yml + rerank.gpu.yml"
	@MODEL_NAME=ci-validate MODEL_SOURCE=hf://ci/repo \
		docker compose -f deploy/compose/rerank.yml -f deploy/compose/rerank.gpu.yml \
			--env-file deploy/compose/.env.example config -q
	@echo "==> deploy/compose/rerank.yml + rerank.integration.yml"
	@MODEL_NAME=ci-validate MODEL_SOURCE=hf://ci/repo \
		docker compose -f deploy/compose/rerank.yml -f deploy/compose/rerank.integration.yml \
			--env-file deploy/compose/.env.example config -q

k8s-validate: ## Validate deploy/k8s/<engine>/ with kubeconform (matches CI; offline)
	@# CI uses kubeconform (offline, schema-bundled) because kubectl 1.27+
	@# needs a live apiserver even for --dry-run=client. Match that here so
	@# local validation and the gate agree. CI pins v0.6.7 / k8s 1.30.5.
	@command -v kubeconform >/dev/null 2>&1 || { \
		echo "kubeconform not found."; \
		echo "  Install (matches CI pin v0.6.7):"; \
		echo "    https://github.com/yannh/kubeconform/releases"; \
		exit 1; \
	}
	@for d in deploy/k8s/*/; do \
		echo "==> $$d"; \
		kubeconform -summary -strict -kubernetes-version 1.30.5 $$d || exit 1; \
	done

vulncheck: ## govulncheck ./... (installs binary on first run)
	@# Pin govulncheck so a future major release can't break this gate
	@# without a deliberate bump here. Keep in sync with .github/workflows/ci.yml.
	@tool="$$(command -v govulncheck || true)"; \
	if [ -z "$$tool" ]; then \
		tool="$$($(GO) env GOPATH)/bin/govulncheck"; \
		GOBIN="$$(dirname "$$tool")" $(GO) install golang.org/x/vuln/cmd/govulncheck@v1.3.0; \
	fi; \
	"$$tool" ./...

clean: ## Remove build artifacts
	rm -rf $(BIN_DIR) coverage.out coverage.html

tidy: ## Run go mod tidy
	$(GO) mod tidy

help: ## Show this help
	@awk 'BEGIN{FS=":.*## "}/^[a-zA-Z_-]+:.*## /{printf "  %-14s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
