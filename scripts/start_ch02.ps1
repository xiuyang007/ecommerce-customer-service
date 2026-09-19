param(
    [int]$Port = 8767
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$DockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($DockerCommand) {
    $DockerPath = $DockerCommand.Source
} else {
    $DockerPath = (Get-Item 'C:\Program Files\Docker\Docker\resources\bin\docker.exe' -ErrorAction Stop).FullName
}

Set-Location $Root
& $DockerPath compose up -d mysql
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    & $DockerPath compose exec -T mysql mysqladmin ping -h 127.0.0.1 -papp_password --silent 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw 'mysql did not become ready' }

python "$Root\scripts\init_db.py"
if ($LASTEXITCODE -ne 0) { throw 'database initialization failed' }

Write-Host "MySQL ready. Start the API with:" -ForegroundColor Green
Write-Host "python -m uvicorn app.main:app --host 127.0.0.1 --port $Port"
