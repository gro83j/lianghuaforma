$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:BASE_DIR = $repoRoot
$env:OUT_DIR = $repoRoot
$env:ARTIFACTS_DIR = Join-Path $repoRoot 'artifacts\019'
$env:REPORT_PATH = Join-Path $repoRoot 'reports\019.txt'
$env:DATA_DIR = Join-Path $repoRoot 'data\42swam'
$env:REF_CFG_PATH = Join-Path $repoRoot 'configs\016_selected_config.json'
$env:THRESHOLDS_016_PATH = Join-Path $repoRoot 'configs\thresholds_walkforward.json'
python (Join-Path $repoRoot 'experiments\20260112_019_Mode4_TP2_ConfidenceMax.py')
