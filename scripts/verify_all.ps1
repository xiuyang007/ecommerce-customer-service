param(
    [int]$Port = 8767
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BaseUrl = "http://127.0.0.1:$Port"
$Server = $null

try {
    Write-Host '[1/6] 编译检查'
    python -m compileall -q "$Root\app" "$Root\tests" "$Root\scripts"
    if ($LASTEXITCODE -ne 0) { throw 'compileall failed' }

    Write-Host '[2/6] 自动化测试'
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'pytest failed' }

    Write-Host '[3/6] 浏览器脚本语法检查'
    node --input-type=module -e "import fs from 'fs'; const html=fs.readFileSync(process.argv[1],'utf8'); const script=html.split('<script>')[1].split('</script>')[0]; new Function(script); console.log('browser_js_syntax_ok')" "$Root\web\index.html"
    if ($LASTEXITCODE -ne 0) { throw 'browser JavaScript syntax check failed' }

    Write-Host '[4/6] 启动 MySQL 并初始化四张表'
    & "$PSScriptRoot\start_ch02.ps1" -Port $Port
    if ($LASTEXITCODE -ne 0) { throw 'MySQL startup failed' }

    Write-Host '[5/6] 启动临时服务'
    $Server = Start-Process -FilePath 'python' `
        -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $health = Invoke-WebRequest -UseBasicParsing "$BaseUrl/health"
            if ($health.StatusCode -eq 200) { $ready = $true; break }
        }
        catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "service did not become ready at $BaseUrl" }

    Write-Host '[6/6] 执行真实 DeepSeek API 与工具链冒烟测试'
    python "$Root\scripts\smoke_api.py" --base-url $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw 'API smoke test failed' }

    python "$Root\scripts\smoke_tools.py" --base-url $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw 'tool smoke test failed' }

    Write-Host 'ALL_CHECKS_PASSED' -ForegroundColor Green
}
finally {
    if ($null -ne $Server) {
        Stop-Process -Id $Server.Id -Force -ErrorAction SilentlyContinue
    }
}
