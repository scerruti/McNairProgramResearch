# Run VADAR on LEGO benchmark (Windows)
# Usage:
#   .\run_lego.ps1                          # Full benchmark (1100 Qs)
#   .\run_lego.ps1 -Lite                    # Lite subset (220 Qs)
#   .\run_lego.ps1 -MaxQuestions 20         # Small test (20 Qs)
#   .\run_lego.ps1 -MaxQuestions 5 -Stub    # Pipeline test (no models)
param(
    [int]$MaxQuestions = -1,
    [switch]$Lite,
    [switch]$Stub
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Verify CUDA before starting
Write-Host "Checking CUDA..."
py -3.11 -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available! Need NVIDIA GPU.'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) { Write-Error "CUDA check failed. Make sure NVIDIA drivers and CUDA toolkit are installed." }

# Verify API key
if (-not (Test-Path "api.key")) { Write-Error "api.key not found! Create it with your OpenAI API key." }

# Build arguments
$args_list = @()
if ($MaxQuestions -gt 0) { $args_list += "--max-questions", $MaxQuestions }
if ($Lite)               { $args_list += "--lite" }
if ($Stub)               { $args_list += "--stub" }

Write-Host "`nRunning VADAR on LEGO benchmark..."
Write-Host "Arguments: $($args_list -join ' ')"
Write-Host ""

py -3.11 run_lego.py @args_list
