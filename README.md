# 电商智能客服 · Chapter 01 纯对话 MVP

先跑通不带工具调用和 Agent 循环的纯对话：

- FastAPI 提供 SSE 流式多轮对话；
- `ChatPromptTemplate` 管理客服 System Prompt；
- `with_structured_output` 从售后描述提取固定字段；
- 内存会话配合保守 token 预算裁剪历史；
- 模型走 OpenAI 兼容协议，地址、模型名、密钥都放在 `.env`。

本章不查询真实订单，不执行退款/换货，也不做工具调用、Agent、RAG、数据库或鉴权。内存会话仅用于本地演示，进程重启后丢失。

## 快速开始

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在 `.env` 中填写 OpenAI 兼容上游：

```env
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_STRUCTURED_METHOD=function_calling
```

浏览器打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 。

## 接口

### `POST /api/v1/chat/stream`

```json
{"session_id": null, "message": "我买的耳机左边没有声音，想换货。"}
```

SSE 事件：`meta`、`delta`、`done`、`error`。

PowerShell 示例：

```powershell
$body = @{ message = '我买的耳机左边没有声音，想换货。' } | ConvertTo-Json -Compress
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/chat/stream `
  -H 'Content-Type: application/json' `
  --data-raw $body
```

也可以直接跑演示脚本：

```powershell
.\scripts\demo_chat.ps1 -Message '我买的耳机左边没有声音，想换货。'
```

第二轮带上上一轮 `session_id`：

```powershell
.\scripts\demo_chat.ps1 -SessionId 's_...' -Message '请记住我刚才的换货诉求。'
```

### `POST /api/v1/after-sales/extract`

返回固定字段：`order_id`、`request_type`、`expected_solution`。

```powershell
.\scripts\demo_after_sales.ps1
```

## 验证

确定性测试（不调用真实模型）：

```powershell
python -m compileall -q app tests scripts
python -m pytest -q
```

真实上游冒烟（需要已配置 `.env` 并启动服务）：

```powershell
python .\scripts\smoke_api.py --base-url http://127.0.0.1:8000
python .\scripts\eval_after_sales.py
python .\scripts\eval_chat.py
```

一键检查会编译、跑 pytest、检查浏览器脚本语法，再临时启动服务做真实 API 冒烟：

```powershell
.\scripts\verify_all.ps1
```

## 目录

- `app/`：FastAPI 入口、Prompt、会话与对话服务
- `web/index.html`：对话和售后提取页面
- `evals/`：售后提取与客服行为评估样例
- `dev-notes/ch01.md`：第一章开发过程记录