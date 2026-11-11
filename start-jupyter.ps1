$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:JUPYTER_RUNTIME_DIR = "$PSScriptRoot\.local\jupyter\runtime"
$env:JUPYTER_CONFIG_DIR = "$PSScriptRoot\.local\jupyter\config"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m jupyterlab --notebook-dir="$PSScriptRoot" --ip=127.0.0.1
