# Fault-injection HTTP server for the integration download-fault matrix
# (tests/integration/download + tests/integration/ollamaurl). Builds the
# standalone faultsrv CLI from tests/download/cmd/faultsrv into a
# distroless/static image so the compose stacks can serve a model URL
# that returns 404/503, drops mid-stream, rotates ETags, etc.
#
# Build from the repo root so the Go module is in context:
#   docker build -f tests/integration/faultsrv/Dockerfile -t llm-init-faultsrv:integration .
#
# The compose files pass `-addr 0.0.0.0:9000` so the server is reachable
# from the llm-init container over the compose network.

FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/faultsrv ./tests/download/cmd/faultsrv

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/faultsrv /usr/local/bin/faultsrv
EXPOSE 9000
ENTRYPOINT ["/usr/local/bin/faultsrv"]
