# PowerShell port of run.sh. Same scenario table, same freshness rules.
# Use this from PowerShell or pwsh; use run.sh from bash / WSL / git-bash.
#
# Usage:
#   .\examples\local\run.ps1 <ID> [up|down|logs]
#
#   ID    : A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 F1
#           (D2 has no env file; see D2-source-mismatch.md)
#   verb  : up      (default) -- docker compose up with the right file set
#           down            -- docker compose down -v
#           logs            -- docker compose logs -f
#
# Environment overrides:
#   $env:LLM_INIT_IMAGE   default: llm-init:local, auto-built on demand
#                         from the current source tree. Set to a
#                         different tag to point at an image you
#                         manage yourself; run.ps1 will then refuse
#                         to start if it is missing instead of
#                         building one for you.
#   $env:SKIP_REBUILD     set to "1" to skip the auto-build even when
#                         the image is missing or stale (will warn
#                         and continue with whatever is on disk).
#   $env:LLM_INIT_PORT    pin the host-side port for llm-init. Default
#                         comes from the scenario's .env (8080). When
#                         that port is already bound on the host, run.ps1
#                         auto-picks a free one in 28080..28099 and
#                         falls back to a random 30000..60000 ephemeral
#                         port; the banner prints the effective URL.
#   $env:USE_GPU          NVIDIA passthrough for llamacpp / ollama
#                         scenarios. Tri-state:
#                            unset (default) -> auto: layer the GPU
#                                  override when `nvidia-smi -L`
#                                  succeeds on the host.
#                            '1'             -> force on (skip host
#                                  probe; useful when nvidia-smi is
#                                  not on PATH but Docker still has
#                                  device access).
#                            '0'             -> force off (CPU mode,
#                                  matches the upstream defaults in
#                                  deploy/compose/*).
#                         vllm/sglang already require GPUs in their
#                         base compose files and ignore this flag.
#
# Auto-build on `up`:
#   Before bringing the stack up, run.ps1 decides whether the default
#   image llm-init:local needs rebuilding (image missing / commit
#   label != HEAD / uncommitted edits under cmd/, internal/, go.mod,
#   go.sum, Dockerfile / -dirty version label). On any of those it
#   runs `docker build` against the repo root with VERSION/COMMIT/
#   BUILD_TIME args matching Makefile. Docker's own layer cache
#   decides how much work that actually does.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Id,

    [Parameter(Position=1)]
    [ValidateSet('up','down','logs')]
    [string]$Verb = 'up'
)

function Write-Stderr($Message) { [Console]::Error.WriteLine($Message) }

$Here = Split-Path -Parent $PSCommandPath
$Root = (Resolve-Path (Join-Path $Here '..\..')).Path

# Every scenario layers a spec override that mounts the right full spec
# (chat / embedding / thinking / translate) and points MODEL_SPEC_PATH at it, so the
# mounted file wins over the MODEL_MODE env seed set in each .env.
$specChat  = '_spec-chat.compose.yml'
$specEmbed = '_spec-embedding.compose.yml'
$specThink = '_spec-thinking.compose.yml'
$specTranslate = '_spec-translate.compose.yml'

$scenarios = @{
    'A1' = @{ engine = 'ollama';   envFile = 'A1-ollama-name.env';       override = $specChat  }
    'A2' = @{ engine = 'llamacpp'; envFile = 'A2-llamacpp-hf-gguf.env';  override = $specChat  }
    'A3' = @{ engine = 'vllm';     envFile = 'A3-vllm-hf-repo.env';      override = $specChat  }
    'A4' = @{ engine = 'sglang';   envFile = 'A4-sglang-hf-repo.env';    override = $specChat  }
    'B1' = @{ engine = 'ollama';   envFile = 'B1-ollama-hf-gguf.env';    override = $specChat  }
    'B2' = @{ engine = 'ollama';   envFile = 'B2-ollama-url.env';        override = $specChat  }
    'B3' = @{ engine = 'llamacpp'; envFile = 'B3-llamacpp-url.env';      override = $specChat  }
    'C1' = @{ engine = 'ollama';   envFile = 'C1-ollama-embedding.env';  override = $specEmbed }
    'C2' = @{ engine = 'llamacpp'; envFile = 'C2-llamacpp-thinking.env'; override = $specThink }
    'D1' = @{ engine = 'llamacpp'; envFile = 'D1-bad-repo.env';          override = $specChat  }
    'D3' = @{ engine = 'llamacpp'; envFile = 'D3-hf-mirror.env';         override = $specChat  }
    'E1' = @{ engine = 'llamacpp'; envFile = 'E1-multi-source.env';      override = $specChat  }
    'E2' = @{ engine = 'ollama';   envFile = 'E2-qwen35-vlm.env';        override = $specChat  }
    'F1' = @{ engine = 'llamacpp'; compose = 'translate'; envFile = 'F1-llamacpp-translate.env'; override = $specTranslate }
}

$key = $Id.ToUpperInvariant()
if ($key -eq 'D2') {
    Write-Stderr "D2 is a manual two-step procedure. Read: $Here\D2-source-mismatch.md"
    exit 2
}
if (-not $scenarios.ContainsKey($key)) {
    Write-Stderr "unknown scenario: $Id (expected A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 F1)"
    exit 2
}

$s       = $scenarios[$key]
$envFile = Join-Path $Here $s.envFile
$composeFile = if ($s.ContainsKey('compose')) { $s.compose } else { $s.engine }
$base    = Join-Path $Root "deploy\compose\$composeFile.yml"
$project = "llmlocal-$($key.ToLower())"

if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Stderr "missing env file: $envFile"
    exit 1
}
if (-not (Test-Path -LiteralPath $base)) {
    Write-Stderr "missing base compose file: $base"
    exit 1
}

$composeArgs = @('-f', $base)
if ($s.override) {
    $overridePath = Join-Path $Here $s.override
    if (-not (Test-Path -LiteralPath $overridePath)) {
        Write-Stderr "missing override file: $overridePath"
        exit 1
    }
    $composeArgs += @('-f', $overridePath)
}

$gpuOverrideName = $null
switch ($s.engine) {
    'llamacpp' { $gpuOverrideName = '_llamacpp-gpu.compose.yml' }
    'ollama'   { $gpuOverrideName = '_ollama-gpu.compose.yml'   }
}

function Test-HostNvidiaAvailable {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    & nvidia-smi -L *> $null
    return ($LASTEXITCODE -eq 0)
}

$gpuOn      = $false
$gpuReason  = $null
if ($gpuOverrideName) {
    $envGpu = $env:USE_GPU
    if ($envGpu -eq '0') {
        $gpuReason = 'off (USE_GPU=0)'
    } elseif ($envGpu -eq '1') {
        $gpuOn = $true
        $gpuReason = 'forced via $env:USE_GPU=1'
    } elseif ([string]::IsNullOrWhiteSpace($envGpu)) {
        if (Test-HostNvidiaAvailable) {
            $gpuOn = $true
            $gpuReason = 'NVIDIA auto-detected on host'
        } else {
            $gpuReason = 'off (no nvidia-smi on host; set $env:USE_GPU=1 to force)'
        }
    } else {
        Write-Stderr "warning: ignoring unrecognized USE_GPU=$envGpu (expected 0 | 1 | unset)"
        $gpuReason = "off (unrecognized USE_GPU=$envGpu)"
    }

    if ($gpuOn) {
        $gpuOverride = Join-Path $Here $gpuOverrideName
        if (-not (Test-Path -LiteralPath $gpuOverride)) {
            Write-Stderr "missing GPU override file: $gpuOverride"
            exit 1
        }
        $composeArgs += @('-f', $gpuOverride)
    }
}

$composeArgs += @('--env-file', $envFile, '-p', $project)

# Resolve effective image: parent env > .env file line. F1 uses
# TRANSLATE_LLM_INIT_IMAGE so the generic LLM_INIT_IMAGE cannot override it.
$imageVariable = if ($key -eq 'F1') { 'TRANSLATE_LLM_INIT_IMAGE' } else { 'LLM_INIT_IMAGE' }
$resolvedImage = [Environment]::GetEnvironmentVariable($imageVariable)
if ([string]::IsNullOrWhiteSpace($resolvedImage)) {
    $line = Select-String -Path $envFile -Pattern "^$imageVariable=" | Select-Object -First 1
    if ($line) {
        $resolvedImage = ($line.Line -replace "^$imageVariable=", '').Trim()
    }
}

function Test-PortFree([int]$Port) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { try { $listener.Stop() } catch {} }
    }
}

function Resolve-HostPort {
    $desired = $env:LLM_INIT_PORT
    if ([string]::IsNullOrWhiteSpace($desired)) {
        $line = Select-String -Path $envFile -Pattern '^LLM_INIT_PORT=' | Select-Object -First 1
        if ($line) { $desired = ($line.Line -replace '^LLM_INIT_PORT=', '').Trim() }
    }
    if ([string]::IsNullOrWhiteSpace($desired)) { $desired = '8080' }
    $desiredInt = [int]$desired

    if (Test-PortFree $desiredInt) { return $desiredInt }

    Write-Stderr ">> host port $desiredInt is in use; picking a free one..."
    $candidates = @()
    foreach ($c in 28080..28099) { $candidates += $c }
    foreach ($i in 1..40)        { $candidates += (Get-Random -Minimum 30000 -Maximum 60000) }
    foreach ($c in $candidates) {
        if (Test-PortFree $c) {
            Write-Stderr ">> using LLM_INIT_PORT=$c (override with `$env:LLM_INIT_PORT=...)"
            return $c
        }
    }
    Write-Stderr "error: could not find a free TCP port"
    exit 1
}

$effectivePort = $null
if ($Verb -eq 'up') {
    $effectivePort = Resolve-HostPort
    $env:LLM_INIT_PORT = "$effectivePort"
}

function Invoke-BuildIfNeeded {
    if ($Verb -ne 'up') { return }

    docker image inspect $resolvedImage *> $null
    $imageExists = ($LASTEXITCODE -eq 0)

    # A custom LLM_INIT_IMAGE tag belongs to the operator. We do not
    # rebuild it; we only refuse to start when it is missing.
    if ($resolvedImage -ne 'llm-init:local') {
        if (-not $imageExists) {
            Write-Stderr @"
error: image '$resolvedImage' not present locally.
       `$env:$imageVariable is set to a non-default tag, so run.ps1
       will not build it for you. Build it yourself, or unset
       `$env:$imageVariable to fall back to the auto-built llm-init:local.
"@
            exit 1
        }
        return
    }

    $reason = $null
    if (-not $imageExists) {
        $reason = "image '$resolvedImage' is not present locally"
    } elseif (Get-Command git -ErrorAction SilentlyContinue) {
        git -C $Root rev-parse --git-dir *> $null
        if ($LASTEXITCODE -eq 0) {
            # Pull labels as JSON to dodge PowerShell's mangling of
            # embedded double quotes inside `{{index .Config.Labels "..."}}`.
            $labelsJson = docker image inspect $resolvedImage -f '{{json .Config.Labels}}' 2>$null
            $labels = $null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($labelsJson) -and $labelsJson -ne 'null') {
                try { $labels = $labelsJson | ConvertFrom-Json } catch { $labels = $null }
            }
            $imgCommit  = if ($labels) { $labels.'org.opencontainers.image.revision' } else { $null }
            $imgVersion = if ($labels) { $labels.'org.opencontainers.image.version' }  else { $null }
            $curCommit  = (git -C $Root rev-parse --short HEAD).Trim()
            $dirtyRaw   = git -C $Root status --porcelain -- cmd internal go.mod go.sum Dockerfile 2>$null
            $dirtyFiles = if ($dirtyRaw) { ($dirtyRaw -join "`n").Trim() } else { '' }

            if ([string]::IsNullOrWhiteSpace($imgCommit)) {
                $reason = "image has no revision label (built outside docker build?)"
            } elseif ($imgCommit.Trim() -ne $curCommit) {
                $reason = "image commit $($imgCommit.Trim()) != current HEAD $curCommit"
            } elseif (-not [string]::IsNullOrEmpty($dirtyFiles)) {
                $reason = "working tree dirty under cmd/ internal/ go.mod go.sum Dockerfile"
            } elseif ($imgVersion -and $imgVersion.Trim().EndsWith('-dirty')) {
                $reason = "last build was from a dirty tree (version=$($imgVersion.Trim()))"
            }
        }
    }

    if (-not $reason) { return }

    if ($env:SKIP_REBUILD -eq '1') {
        Write-Stderr ">> $reason"
        Write-Stderr ">> `$env:SKIP_REBUILD = '1'; continuing with the existing image"
        return
    }

    Write-Stderr ">> $reason"
    Write-Stderr ">> auto-building (set `$env:SKIP_REBUILD = '1' to bypass)"

    $version = (& git -C $Root describe --tags --always --dirty 2>$null)
    if ([string]::IsNullOrWhiteSpace($version)) { $version = 'dev' }
    $commit  = (& git -C $Root rev-parse --short HEAD 2>$null)
    if ([string]::IsNullOrWhiteSpace($commit))  { $commit  = 'none' }
    $buildTime = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')

    Write-Stderr ">> docker build -t $resolvedImage  (VERSION=$($version.Trim()) COMMIT=$($commit.Trim()))"
    # Inline so docker build's stdout streams to the host and $LASTEXITCODE
    # is the real build exit code -- not an array mashed up with build logs.
    docker build `
        --build-arg "VERSION=$($version.Trim())" `
        --build-arg "COMMIT=$($commit.Trim())" `
        --build-arg "BUILD_TIME=$buildTime" `
        -t $resolvedImage `
        $Root
    if ($LASTEXITCODE -ne 0) {
        Write-Stderr ">> docker build failed (exit $LASTEXITCODE); aborting"
        exit 1
    }
}

Invoke-BuildIfNeeded

Write-Host "scenario : $key"
Write-Host "engine   : $($s.engine)"
Write-Host "project  : $project"
Write-Host "env-file : $envFile"
if ($s.override) { Write-Host "override : $(Join-Path $Here $s.override)" }
Write-Host "image    : $resolvedImage"
if ($gpuOn) {
    if ($s.engine -eq 'llamacpp') { Write-Host "gpu      : on  (:server-cuda; $gpuReason)" }
    else                          { Write-Host "gpu      : on  ($gpuReason)" }
} elseif ($gpuOverrideName) {
    Write-Host "gpu      : $gpuReason"
}
if ($effectivePort) { Write-Host "url      : http://localhost:$effectivePort" }
Write-Host ""

switch ($Verb) {
    'up'   { docker compose @composeArgs up }
    'down' { docker compose @composeArgs down -v }
    'logs' { docker compose @composeArgs logs -f }
}
exit $LASTEXITCODE
