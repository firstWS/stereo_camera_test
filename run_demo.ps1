$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

. (Join-Path $RepoRoot "scripts\sync-session-path.ps1")

$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Virtual env missing. Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
& $Py "$RepoRoot\experiments\repeatability_run.py" --config "$RepoRoot\configs\demo.yaml"
