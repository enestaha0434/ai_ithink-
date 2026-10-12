<#
.SYNOPSIS
  HyperAI'yi Windows Python PATH ayarına bağlı kalmadan başlatır.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [ValidateRange(1, 4096)]
    [int]$MaxNewTokens = 128,
    [ValidateRange(0.01, 10.0)]
    [double]$Temperature = 0.8,
    [ValidateRange(0.01, 1.0)]
    [double]$TopP = 0.95
)

$ErrorActionPreference = "Stop"
$pythonCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) {
    throw "Çalışan Python kurulumu bulunamadı. Python'u https://www.python.org/downloads/ adresinden kurun."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot
try {
    & $python main.py --checkpoint $Checkpoint --max-new-tokens $MaxNewTokens --temperature $Temperature --top-p $TopP
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
