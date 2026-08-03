[CmdletBinding()]
param(
    [string]$LingBotRepository = "robbyant/lingbot-vla-v2-6b"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Download LingBot-VLA 2.0 base models ==="

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "The supported Hugging Face CLI, hf, was not found. Install it first: pip install -U huggingface_hub"
}

$modelsDir = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

function Download-Model {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$Directory,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    Write-Host "Downloading $DisplayName..."
    & hf download $Repository --local-dir $Directory
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download $DisplayName. Exit code: $LASTEXITCODE"
    }
}

function Download-ModelFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$RemoteFile,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    $destinationDir = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
    Write-Host "Downloading $DisplayName..."
    & hf download $Repository $RemoteFile --local-dir $destinationDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download $DisplayName. Exit code: $LASTEXITCODE"
    }

    $downloadedFile = Join-Path $destinationDir $RemoteFile
    if (-not (Test-Path $downloadedFile)) {
        throw "Expected downloaded file was not found: $downloadedFile"
    }
    if ($downloadedFile -ne $Destination) {
        Move-Item -Path $downloadedFile -Destination $Destination -Force
    }
}

# 1. Qwen3-VL-4B-Instruct (vision-language model base)
Download-Model "Qwen/Qwen3-VL-4B-Instruct" (Join-Path $modelsDir "Qwen3-VL-4B-Instruct") "Qwen3-VL-4B-Instruct"

# 2. LingBot-VLA v2.6B
Download-Model $LingBotRepository (Join-Path $modelsDir "lingbot-vla-v2-6b") "LingBot-VLA v2.6B"

# 3. MoGe-2 depth checkpoint. LingBot-Depth is included in the LingBot-VLA repository.
$depthDir = Join-Path $modelsDir "MoRGBD"
Download-ModelFile "Ruicheng/moge-2-vitb-normal" "model.pt" (Join-Path $depthDir "moge2-vitb-normal.pt") "MoGe-2 ViT-B Normal"

# 4. Video-DINO teacher checkpoint/config are included in the LingBot-VLA repository.

Write-Host "All base models were downloaded to $modelsDir"
Write-Host "Override the LingBot checkpoint repository with -LingBotRepository if needed."