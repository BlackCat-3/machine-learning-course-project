param(
    [string]$OutputRoot = 'outputs/final_models',
    [string]$Submission = 'outputs/submission.csv'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup.ps1 first.' }

function Invoke-TaskPython {
    param([string[]]$CommandArgs)
    & $pythonPath @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw ('Python command failed: ' + ($CommandArgs -join ' ')) }
}

$baselineRun = Join-Path $OutputRoot 'baseline'
$mixupRun = Join-Path $OutputRoot 'mixup'
# Keep the 18-epoch cosine schedule, but stop at the epochs selected on holdout.
Invoke-TaskPython -CommandArgs @(
    '-u','train.py','--data','data/cache.npz','--output-dir',$baselineRun,
    '--epochs','18','--stop-after-epochs','7','--full-train',
    '--batch-size','128','--image-size','112','--lr','0.0003',
    '--weight-decay','0.0001','--label-smoothing','0.05','--dropout','0.2',
    '--seed','42','--device','cuda'
)
Invoke-TaskPython -CommandArgs @(
    '-u','train.py','--data','data/cache.npz','--output-dir',$mixupRun,
    '--epochs','18','--stop-after-epochs','12','--full-train',
    '--batch-size','128','--image-size','112','--lr','0.0003',
    '--weight-decay','0.001','--label-smoothing','0.05','--dropout','0.3',
    '--seed','43','--mixup-alpha','0.2','--mixup-prob','1','--device','cuda'
)
Invoke-TaskPython -CommandArgs @(
    '-u','predict.py','--data','data/cache.npz',
    '--checkpoints',(Join-Path $baselineRun 'final.pt'),(Join-Path $mixupRun 'final.pt'),
    '--tta','--device','cuda','--output',$Submission
)
Invoke-TaskPython -CommandArgs @(
    'verify_submission.py',$Submission,'--json',([IO.Path]::ChangeExtension($Submission,'check.json'))
)
