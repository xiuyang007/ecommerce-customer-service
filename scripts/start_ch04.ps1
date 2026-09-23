param(
    [switch]$Ingest
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
& $DockerPath compose up -d mysql etcd minio milvus
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }

$mysqlReady = $false
$milvusReady = $false
for ($i = 0; $i -lt 90; $i++) {
    & $DockerPath compose exec -T mysql mysqladmin ping -h 127.0.0.1 -papp_password --silent 2>$null
    if ($LASTEXITCODE -eq 0) { $mysqlReady = $true }
    try {
        $health = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:9091/healthz' -TimeoutSec 2
        if ($health.StatusCode -eq 200) { $milvusReady = $true }
    }
    catch { }
    if ($mysqlReady -and $milvusReady) { break }
    Start-Sleep -Seconds 2
}
if (-not $mysqlReady) { throw 'mysql did not become ready' }
if (-not $milvusReady) { throw 'milvus did not become ready' }

Get-Content -Raw "$Root\scripts\init_ch04.sql" | & $DockerPath compose exec -T mysql mysql -uapp -papp_password -D ecommerce_support
if ($LASTEXITCODE -ne 0) { throw 'chapter 04 DDL failed' }

python "$Root\scripts\init_db.py"
if ($LASTEXITCODE -ne 0) { throw 'database initialization failed' }

if ($Ingest) {
    python "$Root\scripts\ingest_knowledge.py" --recreate
    if ($LASTEXITCODE -ne 0) { throw 'knowledge ingestion failed' }
}

Write-Host 'Chapter 04 infrastructure is ready.' -ForegroundColor Green
