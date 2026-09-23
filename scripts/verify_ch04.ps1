param(
    [int]$Port = 8767,
    [switch]$WithGeneration
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BaseUrl = "http://127.0.0.1:$Port"
$Server = $null

try {
    Write-Host '[1/6] compile'
    python -m compileall -q "$Root\app" "$Root\tests" "$Root\scripts"
    if ($LASTEXITCODE -ne 0) { throw 'compileall failed' }

    Write-Host '[2/6] pytest'
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'pytest failed' }

    Write-Host '[3/6] frontend syntax'
    node --input-type=module -e "import fs from 'fs'; const html=fs.readFileSync(process.argv[1],'utf8'); const script=html.split('<script>')[1].split('</script>')[0]; new Function(script); console.log('browser_js_syntax_ok')" "$Root\web\index.html"
    if ($LASTEXITCODE -ne 0) { throw 'frontend syntax failed' }

    Write-Host '[4/6] start MySQL + Milvus and ingest knowledge'
    & "$PSScriptRoot\start_ch04.ps1" -Ingest
    if ($LASTEXITCODE -ne 0) { throw 'infrastructure startup failed' }

    Write-Host '[5/6] start API and smoke tools'
    $Server = Start-Process -FilePath 'python' `
        -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $health = Invoke-WebRequest -UseBasicParsing "$BaseUrl/health" -TimeoutSec 2
            if ($health.StatusCode -eq 200) { $ready = $true; break }
        }
        catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'API did not become ready' }
    python "$Root\scripts\smoke_api.py" --base-url $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw 'API smoke failed' }
    python "$Root\scripts\smoke_tools.py" --base-url $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw 'tool smoke failed' }
    python "$Root\scripts\smoke_rag.py"
    if ($LASTEXITCODE -ne 0) { throw 'RAG smoke failed' }

    Write-Host '[6/6] four-strategy retrieval report'
    $EvalArgs = @("$Root\scripts\eval_retrieval.py")
    if ($WithGeneration) { $EvalArgs += '--with-generation' }
    python @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw 'retrieval evaluation failed' }

    Write-Host 'ALL_CHAPTER_04_CHECKS_PASSED' -ForegroundColor Green
}
finally {
    if ($null -ne $Server) {
        Stop-Process -Id $Server.Id -Force -ErrorAction SilentlyContinue
    }
}
