#!/usr/bin/env bash
# tools/install-lint.sh -- install the golangci-lint version this repo expects.
#
# Why this script exists
# ----------------------
# CI pins golangci-lint to a specific version (.github/workflows/ci.yml
# uses golangci/golangci-lint-action@v8.0.0 with version: vX.Y.Z). A
# dev machine running a different binary -- typically whatever
# `brew install golangci-lint` resolved last week -- produces a
# different finding set, which usually means CI fails on a PR a dev
# thought was clean.
#
# The Makefile `lint` target used to print "CI pins v1.64.8" in its
# error message, but that copy decayed when CI bumped to v2.12.2 in
# Track F.2 (v1.0.6). This script is the single source of truth: edit
# the pin here, the Makefile call site picks it up automatically, and
# the CHANGELOG-time bump can be done in one file.
#
# Usage
# -----
#   bash tools/install-lint.sh             # installs to $HOME/go/bin
#   GOBIN=/tmp bash tools/install-lint.sh  # respects GOBIN
#
# After running, confirm:
#   golangci-lint version
# matches the pin below (the action dies on any mismatch in CI).
#
# This script is intentionally simple shell, not a Go install helper,
# because v2's release artifacts are released as a single static
# binary and `go install` against the v2 module path is supported but
# brittle across Go toolchain versions. The official upstream install
# script handles all the platform / arch detection that we don't need
# to recreate.
#
# asdf / mise users: prefer `asdf install` / `mise install` driven
# by the .tool-versions file at the repo root. This script targets
# developers who don't run a tool-version manager.

set -euo pipefail

# Single source of truth. Keep in sync with:
#   * .tool-versions
#   * .github/workflows/ci.yml (golangci-lint-action `version:` field)
GOLANGCI_LINT_VERSION="v2.12.2"

# Default install dir is $GOPATH/bin (matches `go install` semantics).
# `go env GOBIN` returns "" when GOBIN is unset; fall through to
# $GOPATH/bin in that case.
target="${GOBIN:-}"
if [ -z "${target}" ]; then
    if command -v go >/dev/null 2>&1; then
        target="$(go env GOPATH)/bin"
    else
        target="${HOME}/go/bin"
    fi
fi
mkdir -p "${target}"

echo "==> installing golangci-lint ${GOLANGCI_LINT_VERSION} into ${target}"

# Upstream-published install script. Pinned to a specific tag rather
# than `master` so a compromised upstream cannot silently swap the
# install logic on us.
#
# Provenance: https://github.com/golangci/golangci-lint/blob/v2.12.2/install.sh
# Documentation: https://golangci-lint.run/usage/install/
curl -sSfL \
    "https://raw.githubusercontent.com/golangci/golangci-lint/${GOLANGCI_LINT_VERSION}/install.sh" \
    | sh -s -- -b "${target}" "${GOLANGCI_LINT_VERSION}"

echo
echo "==> installed:"
"${target}/golangci-lint" version

echo
echo "Add ${target} to your PATH if it is not there already:"
echo "  export PATH=\"${target}:\$PATH\""
