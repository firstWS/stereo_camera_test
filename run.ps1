<#
.SYNOPSIS
  PATH 동기화 → (필요 시) setup.ps1 → 데모 또는 전체 설정으로 파이프라인 실행.

.EXAMPLE
  .\run.ps1
  .\run.ps1 -Setup
  .\run.ps1 -Full
  .\run.ps1 -ImageFolder
  .\run.ps1 -Orbbec
  .\run.ps1 -Config configs\default.yaml
#>
param(
    [switch]$Setup,
    [switch]$Full,
    [switch]$ImageFolder,
    [switch]$Orbbec,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

. (Join-Path $RepoRoot "scripts\sync-session-path.ps1")

if ([string]::IsNullOrWhiteSpace($Config)) {
    if ($ImageFolder) {
        $Config = "configs\image_folder.yaml"
    }
    elseif ($Orbbec) {
        $Config = "configs\orbbec_gemini.yaml"
    }
    elseif ($Full) {
        $Config = "configs\default.yaml"
    }
    else {
        $Config = "configs\demo.yaml"
    }
}

$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$needsSetup = $Setup -or -not (Test-Path -LiteralPath $venvPy)

if ($needsSetup) {
    Write-Host "Running setup (venv / pip / placeholder calibration / smoke test)..." -ForegroundColor Cyan
    & (Join-Path $RepoRoot "setup.ps1")
    if (-not $?) {
        exit 1
    }
}

if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Host "Expected venv at $venvPy but it is missing." -ForegroundColor Red
    exit 1
}

$cfgPath = Join-Path $RepoRoot $Config
if (-not (Test-Path -LiteralPath $cfgPath)) {
    Write-Host "Config not found: $cfgPath" -ForegroundColor Red
    exit 1
}

Write-Host "Running repeatability_run with $Config ..." -ForegroundColor Green
& $venvPy (Join-Path $RepoRoot "experiments\repeatability_run.py") --config $cfgPath
