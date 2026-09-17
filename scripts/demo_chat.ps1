param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$SessionId = '',
    [string]$Message = '你好，请用一句话介绍你能帮助我做什么。'
)

$payload = @{
    session_id = if ($SessionId) { $SessionId } else { $null }
    message = $Message
} | ConvertTo-Json -Compress

curl.exe -sS -N --max-time 90 -X POST "$BaseUrl/api/v1/chat/stream" `
    -H 'Content-Type: application/json' `
    --data-raw $payload

if ($LASTEXITCODE -ne 0) {
    throw "chat request failed with exit code $LASTEXITCODE"
}
