# 电商智能客服系统（Chapter 02）

第一章已跑通纯对话；第二章增加 MySQL/SQLAlchemy 持久化、五个业务工具和单轮 Function Calling，并把这些能力接回同一个 SSE 聊天入口。

## 已实现

- FastAPI `POST /api/v1/chat/stream`：SSE 流式输出，事件包含 `meta`、`delta`、`done`、`error`；
- `ChatPromptTemplate` 管理客服角色和行为约束；
- 内存会话、多轮上下文和保守 token 预算裁剪；
- `POST /api/v1/after-sales/extract`：通过 `with_structured_output` 返回固定售后字段；
- `ChatOpenAI` 使用 `LLM_BASE_URL` 对接 OpenAI-compatible 上游；
- MySQL 四张业务表：`faq`、`conversations`、`messages`、`tickets`；
- 五个 LangChain `@tool`：订单、商品、物流、FAQ 和人工工单；
- 单轮 Function Calling：最多执行一个工具，再把结果回灌给模型生成最终回答；
- 聊天页显示工具轨迹徽章，并提供四张业务表的只读预览页签。

## 安装和配置

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
# 编辑 .env，至少设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
```

示例配置：

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_STRUCTURED_METHOD=json_mode
```

Ollama 的 OpenAI-compatible 端点可按实际本地模型改成 `http://localhost:11434/v1`，密钥通常填写任意非空占位值；是否支持结构化输出要以实际模型/端点能力为准。

## 启动

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

## 功能演示

### 1. SSE 流式对话

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message":"我买的耳机左边没有声音，想换货。"}'
```

从 `meta` 事件中复制 `session_id`，再发第二轮：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"session_id":"s_xxx","message":"订单号是 A202609130001。"}'
```

### 2. 售后描述结构化提取

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/after-sales/extract `
  -H "Content-Type: application/json" `
  -d '{"text":"订单 A202609130001，耳机左边没声音，我想换一个新的。"}'
```

预期响应形状：

```json
{
  "order_id": "A202609130001",
  "request_type": "exchange",
  "expected_solution": "换一个新的耳机"
}
```

## 验证

```powershell
python -m compileall -q app tests
python -m pytest -q
```

当前确定性测试覆盖请求校验、上下文裁剪、会话并发/提交语义、SSE 事件、多轮历史承接、结构化响应和缺少密钥时的错误语义。当前 `.env` 使用 DeepSeek OpenAI 兼容接口，并已完成真实 SSE、多轮上下文、Function Calling、MySQL 落库和结构化 JSON 验收。

## 目录

- `D:\实习\Project01\PROJECT.md`：目标、边界、接口和设计决策；
- `D:\实习\Project01\app\`：应用代码；
- `D:\实习\Project01\tests\`：确定性测试；
- `D:\实习\Project01\evals\`：模型评估样例；
- `D:\实习\Project01\dev-notes\ch01.md`：第一章开发留痕；
- `D:\实习\Project01\dev-notes\ch02.md`：第二章开发留痕。




## PowerShell 请求注意事项

PowerShell 不使用 Linux 的 `\` 作为换行续写符。为避免 `curl.exe` 原生参数的引号解析问题，推荐这样发送 JSON：

```powershell
$body = @{ message = '我买的耳机左边没有声音，想换货。' } | ConvertTo-Json -Compress
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/chat/stream `
  -H 'Content-Type: application/json' `
  --data-raw $body
```

`GET /api/v1/chat/stream` 返回 `405` 是正常的，因为聊天接口只接受 `POST`；浏览器自动请求 `/favicon.ico` 返回 `404` 不影响业务。

### 3. PowerShell 售后结构化提取脚本

```powershell
Set-Location 'D:\实习\Project01'
.\scripts\demo_after_sales.ps1
```

也可以传入自定义描述：

```powershell
.\scripts\demo_after_sales.ps1 -Text '订单 B-9 的衣服破了，我想申请退款。'
```

### 4. 售后评估集回归

服务启动后执行 5 条真实样例评估：

```powershell
python 'D:\实习\Project01\scripts\eval_after_sales.py'
```

该脚本对 `order_id`、`request_type` 做严格字段比对，对自然语言 `expected_solution` 做归一化和意图级比对；评估集文件为 `D:\实习\Project01\evals\after_sales_examples.jsonl`。



## 浏览器使用

服务启动后打开：

```text
http://127.0.0.1:8000/
```

浏览器页面位于 `D:\实习\Project01\web\index.html`，已支持：

- 输入框和发送按钮；
- SSE 增量回复；
- 当前会话 `session_id` 保存在浏览器 `localStorage`；
- “新建会话”按钮；
- Enter 发送、Shift + Enter 换行；
- 移动端自适应布局。

聊天页连接客服 SSE 接口；工具调用只使用演示数据或本地 FAQ/工单表，不会操作真实订单。

### 浏览器中的售后提取

打开页面后切换到“售后提取”标签，可以直接调用 `/api/v1/after-sales/extract`，并查看：

- 订单号；
- 诉求类型；
- 期望方案。

该功能与对话模式共用当前服务和 `.env` 配置。

### 浏览器中的数据库表

打开页面后切换到“数据表”页签，可以只读查看 `faq`、`conversations`、`messages`、`tickets` 的表结构、总行数和最近记录。

### 5. 客服对话行为评估

服务启动后执行多轮真实模型评估：

```powershell
python 'D:\实习\Project01\scripts\eval_chat.py'
```

评估内容包括上下文承接、售后信息追问、禁止虚假声称已经退款/换货，以及未知店铺政策不编造。样例文件为 `D:\实习\Project01\evals\chat_examples.jsonl`。

### 6. PowerShell 对话演示脚本

直接运行，不要把说明性中文句子当作 PowerShell 命令输入：

```powershell
& 'D:\实习\Project01\scripts\demo_chat.ps1' `
  -Message '我买的耳机左边没有声音，想换货。'
```

复用第二轮上下文：

```powershell
& 'D:\实习\Project01\scripts\demo_chat.ps1' `
  -SessionId '上一轮 meta 事件返回的 session_id' `
  -Message '请记住我刚才的换货诉求。'
```

### 浏览器聊天记录恢复

浏览器会把当前页面的聊天展示记录保存在 `localStorage`，刷新页面后可以恢复可见记录；服务端会话仍然由 `session_id` 管理，服务重启或会话过期后不会伪造历史上下文。点击“新建会话”会同时清理本地展示记录。

### 安全回归样例

当前 `D:\实习\Project01\evals\chat_examples.jsonl` 包含 5 组真实对话样例，新增覆盖：

- Prompt 注入时不泄露 System Prompt；
- 不把用户要求直接当作退款已完成；
- 不把未查询的订单状态说成已发货。

最新真实评估结果：`5/5` cases、`9/9` assertions 通过。

### 7. 一键全自动验证

无需手动启动服务、复制 `session_id` 或切换终端：

```powershell
& 'D:\实习\Project01\scripts\verify_all.ps1'
```

它会自动完成：编译检查、`pytest`、浏览器 JavaScript 语法检查、启动 MySQL、初始化四张表、临时启动 Uvicorn、真实 DeepSeek SSE、多轮上下文、Function Calling 和售后结构化提取验证。默认使用临时端口 `8767`，验证结束后自动停止临时 API 服务。

## Chapter 02: Function Calling

第二章在现有聊天入口上增加一次性工具调用。MySQL 通过 Docker Compose 启动，数据保存在命名卷中：

```powershell
Set-Location 'D:\实习\Project01'
.\scripts\start_ch02.ps1
```

该脚本会启动 MySQL 8.4、等待健康检查，并执行 `scripts/init_db.py` 创建四张表和 FAQ 测试数据。随后启动 API：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8767
```

第二章验收问法：

```text
订单 1001 的物流到哪了
退货政策是什么
邮费是多少
```

真实模型工具链验证：

```powershell
python '.\scripts\smoke_tools.py' --base-url http://127.0.0.1:8767
```

`邮费是多少` 当前预期由 `query_faq` 返回空结果，这是本章刻意保留的漏召回样例。
