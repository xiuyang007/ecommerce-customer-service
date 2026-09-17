param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$Text = '订单 A202609130001，耳机左边没有声音，我想换一个新的。'
)

# Keep Chinese JSON readable in Windows PowerShell / Windows Terminal.
if ($env:OS -eq 'Windows_NT') {
    & cmd.exe /c chcp 65001 > $null
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$body = @{ text = $Text } | ConvertTo-Json -Compress
$uri = "$BaseUrl/api/v1/after-sales/extract"

try {
    $response = Invoke-RestMethod -Uri $uri -Method Post `
        -ContentType 'application/json; charset=utf-8' `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
    [Console]::WriteLine(($response | ConvertTo-Json -Depth 10 -Compress))
}
catch {
    throw "after-sales request failed: $($_.Exception.Message)"
}
