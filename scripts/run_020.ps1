$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:BASE_DIR = $repoRoot
$env:OUT_DIR = $repoRoot
$env:BASELINE_019_TXT = Join-Path $repoRoot 'reports\019.txt'
$env:BASELINE_019_ART = Join-Path $repoRoot 'artifacts\019'
$env:SCRIPT_019 = Join-Path $repoRoot 'experiments\20260112_019_Mode4_TP2_ConfidenceMax.py'
$env:REPORT_020 = Join-Path $repoRoot 'reports\020.txt'
$env:ART_020 = Join-Path $repoRoot 'artifacts\020'
python (Join-Path $repoRoot 'experiments\20260112_020_Mode4_StabilityAudit.py')
