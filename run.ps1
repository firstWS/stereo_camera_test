<#
.SYNOPSIS
  PATH 동기화 → (필요 시) setup.ps1 → 데모 또는 전체 설정으로 파이프라인 실행.

.EXAMPLE
  .\run.ps1
  .\run.ps1 -Setup
  .\run.ps1 -Full
  .\run.ps1 -ImageFolder
  .\run.ps1 -Orbbec
  .\run.ps1 -Orbbec -RegisterObjectAnchor
  .\run.ps1 -Orbbec -Capture -Positive
  .\run.ps1 -Orbbec -Capture -Negative -CaptureCount 200 -CaptureInterval 0.5
  .\run.ps1 -Config configs\default.yaml
#>
param(
    [switch]$Setup,
    [switch]$Full,
    [switch]$ImageFolder,
    [switch]$Orbbec,
    [switch]$RegisterObjectAnchor,
    [switch]$Capture,
    [switch]$Positive,
    [switch]$Negative,
    [int]$CaptureCount = 100,
    [double]$CaptureInterval = 1.0,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$captureUsage = @"
Valid capture examples:
  .\run.ps1 -Orbbec -Capture -Positive
  .\run.ps1 -Orbbec -Capture -Negative
  .\run.ps1 -Orbbec -Capture -Positive -CaptureCount 150 -CaptureInterval 0.5
"@

$captureFlagsPresent = (
    $Capture -or $Positive -or $Negative -or
    $PSBoundParameters.ContainsKey("CaptureCount") -or
    $PSBoundParameters.ContainsKey("CaptureInterval")
)
if ($captureFlagsPresent) {
    if (-not $Capture) {
        Write-Host "Capture type flags require -Capture.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
    if (-not $Orbbec) {
        Write-Host "-Capture requires -Orbbec.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
    if ($Positive -eq $Negative) {
        Write-Host "Choose exactly one of -Positive or -Negative.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
    if ($CaptureCount -le 0) {
        Write-Host "-CaptureCount must be greater than zero.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
    if ($CaptureInterval -le 0) {
        Write-Host "-CaptureInterval must be greater than zero.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
    if ($RegisterObjectAnchor) {
        Write-Host "-RegisterObjectAnchor cannot be combined with -Capture.`n$captureUsage" -ForegroundColor Red
        exit 2
    }
}

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
$runArgs = @((Join-Path $RepoRoot "experiments\repeatability_run.py"), "--config", $cfgPath)
if ($RegisterObjectAnchor) {
    $runArgs += "--register-object-anchor"
}
if ($Capture) {
    $runArgs += @(
        "--capture-type", $(if ($Positive) { "positive" } else { "negative" }),
        "--capture-count", $CaptureCount.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--capture-interval", $CaptureInterval.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}
& $venvPy @runArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
