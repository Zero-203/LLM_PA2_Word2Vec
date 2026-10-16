param([string]$PythonPath)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
	$Python = Join-Path $ProjectRoot ".venv39\Scripts\python.exe"
}
else {
	$Python = (Resolve-Path $PythonPath).Path
}
$Smoke = Join-Path $ProjectRoot "scripts\run_small_sample.py"
$PublicMetrics = Join-Path $ProjectRoot "scripts\export_public_metrics.py"
$Generator = Join-Path $ProjectRoot "scripts\build_stage6_deliverables.py"

if (-not (Test-Path $Python)) {
	throw "Python 3.9 virtual environment not found: $Python. Follow README.md to create it and install requirements.txt."
}
if (-not (Test-Path $Generator)) {
	throw "Stage 6 generator not found: $Generator"
}
if (-not (Test-Path $Smoke)) {
	throw "Small-sample pipeline not found: $Smoke"
}
if (-not (Test-Path $PublicMetrics)) {
	throw "Public metric exporter not found: $PublicMetrics"
}

Push-Location $ProjectRoot
try {
	& $Python $Smoke
	if ($LASTEXITCODE -ne 0) {
		throw "Small-sample end-to-end pipeline failed. Python exit code: $LASTEXITCODE"
	}
	& $Python $PublicMetrics
	if ($LASTEXITCODE -ne 0) {
		throw "Public metric export failed. Python exit code: $LASTEXITCODE"
	}
	& $Python $Generator
	if ($LASTEXITCODE -ne 0) {
		throw "Stage 6 deliverable generation failed. Python exit code: $LASTEXITCODE"
	}
}
finally {
	Pop-Location
}
