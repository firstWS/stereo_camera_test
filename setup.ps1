$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

. (Join-Path $RepoRoot "scripts\sync-session-path.ps1")

function Test-PythonExe([string]$ExePath) {
    if (-not $ExePath -or -not (Test-Path -LiteralPath $ExePath)) {
        return $false
    }
    try {
        $ver = & $ExePath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        return ($ver | Select-Object -Last 1).Trim() -eq "3.12"
    }
    catch {
        return $false
    }
}

function Find-PythonExe {
    $candidates = [System.Collections.Generic.List[string]]::new()

    if ($env:STEREO_POC_PYTHON) {
        $candidates.Add($env:STEREO_POC_PYTHON.Trim())
    }

    # User-local installs (any Python3x folder under Programs\Python)
    $pyRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $pyRoot) {
        Get-ChildItem -LiteralPath $pyRoot -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                $exe = Join-Path $_.FullName "python.exe"
                $candidates.Add($exe)
            }
    }

    # python.org per-machine installs (e.g. C:\Program Files\Python312)
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $root -or -not (Test-Path -LiteralPath $root)) { continue }
        Get-ChildItem -LiteralPath $root -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
            ForEach-Object {
                $exe = Join-Path $_.FullName "python.exe"
                $candidates.Add($exe)
            }
    }

    # Explicit version folders (common python.org layout)
    foreach ($ver in @(312)) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$ver\python.exe"
        $candidates.Add($candidate)
    }

    # PATH via Get-Command / where.exe (skip Microsoft Store stub)
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and ($cmd.Source -notmatch "WindowsApps")) {
            $candidates.Add($cmd.Source)
        }
    }

    try {
        $whereOut = & where.exe python 2>$null
        if ($whereOut) {
            foreach ($line in $whereOut) {
                $line = $line.Trim()
                if ($line -match "WindowsApps") { continue }
                $candidates.Add($line)
            }
        }
    }
    catch {
        # where.exe missing or no matches
    }

    $seen = @{}
    foreach ($exe in $candidates) {
        if (-not $exe) { continue }
        $key = $exe.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        if (Test-PythonExe $exe) {
            return $exe
        }
    }

    return $null
}

$pythonExe = Find-PythonExe
if (-not $pythonExe) {
    Write-Host "Python을 찾지 못했습니다." -ForegroundColor Red
    Write-Host ""
    Write-Host "1) https://www.python.org/downloads/ 에서 설치 후 설치 마법사에서 'Add python.exe to PATH' 체크"
    Write-Host "2) 또는 PowerShell에서 실제 python.exe 경로를 지정:"
    Write-Host '   $env:STEREO_POC_PYTHON = "C:\Path\to\python.exe"'
    Write-Host "   .\setup.ps1"
    Write-Host ""
    Write-Host "3) Microsoft Store용 'python' 스텁만 있으면 제거하거나, 위처럼 별도 설치본 경로를 지정하세요."
    exit 1
}

Write-Host "Using: $pythonExe"
& $pythonExe --version

$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$venvRoot = Join-Path $RepoRoot ".venv"
if (-not (Test-PythonExe $venvPy)) {
    if (Test-Path -LiteralPath $venvRoot) {
        $resolvedVenv = (Resolve-Path -LiteralPath $venvRoot).Path
        if (-not $resolvedVenv.StartsWith($RepoRoot + "\")) {
            throw "Refusing to remove venv outside workspace: $resolvedVenv"
        }
        Write-Host "Removing broken/incompatible venv: $resolvedVenv" -ForegroundColor Yellow
        Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
    }
    & $pythonExe -m venv "$RepoRoot\.venv"
}

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r "$RepoRoot\requirements.txt"

$calibrationPath = Join-Path $RepoRoot "calibration\stereo_calib.yaml"
if (-not (Test-Path -LiteralPath $calibrationPath)) {
    & $venvPy "$RepoRoot\scripts\create_placeholder_calibration.py"
}
else {
    Write-Host "Keeping existing calibration: $calibrationPath"
}
& $venvPy "$RepoRoot\scripts\smoke_test.py"
& $venvPy "$RepoRoot\scripts\verify_environment.py"
& $venvPy -m pytest "$RepoRoot\tests" -q -p no:cacheprovider
& $venvPy "$RepoRoot\scripts\synthetic_object_anchor_test.py"

Write-Host "Setup OK. Live camera demo: .\run_demo.ps1"
