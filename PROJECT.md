# 电商智能客服系统：Chapter 02（累计第一章）

## 目标

先跑通不带工具调用和 Agent 循环的纯对话 MVP：

- FastAPI 提供 SSE 流式多轮对话接口；
- `ChatPromptTemplate` 管理客服 System Prompt；
- `with_structured_output` 提取售后描述中的固定字段；
- 内存会话配合保守 token 预算裁剪历史；
- 模型通过 OpenAI 兼容协议配置，地址、模型名、密钥均来自 `.env`。

## 范围边界

本章不查询真实订单、不执行退款/换货、不做工具调用、Agent 循环、RAG、数据库或鉴权。内存会话用于本地演示，进程重启后丢失，建议单 worker 运行。

## 设计决策

- 使用 `langchain-openai` 的 `ChatOpenAI` 作为统一 OpenAI 协议客户端，通过 `LLM_BASE_URL` 切换兼容上游。
- 对话使用 `StreamingResponse` + `text/event-stream`，事件分为 `meta`、`delta`、`done`、`error`。
- 售后字段使用 Pydantic Schema，并默认用 `with_structured_output(..., method="function_calling")`；实际上游不支持时由配置切换方法，不能将失败伪装成成功。
- 历史只保存成功完成的完整 user/assistant 对，生成失败或断流不污染上下文。
- 裁剪采用保守字符估算，保留完整对话轮次；当前消息超过预算直接拒绝。

## 接口

### `POST /api/v1/chat/stream`

```json
{"session_id": null, "message": "我买的耳机左边没有声音，想换货。"}
```

SSE 事件示例：

```text
event: meta
data: {"session_id":"s_...","request_id":"r_..."}

event: delta
data: {"text":"您好，"}

event: done
data: {"finish_reason":"stop"}
```

### `POST /api/v1/after-sales/extract`

```json
{"text":"订单 A202609130001，耳机左边没声音，我想换一个新的。"}
```

响应固定为 `order_id`、`request_type`、`expected_solution`。

## 当前进度（2026-09-13）

- 设计：完成
- 基础配置和模型客户端：完成
- 对话 SSE：完成
- 内存会话与上下文裁剪：完成
- 售后结构化提取：完成
- 确定性自动化测试：13 passed
- 本地 FastAPI 启动和 `/health` 冒烟：通过
- 真实 GLM-5.3-Flash 上游验证：完成（流式、多轮上下文、结构化 JSON）
- 售后评估集：5/5 结构字段与意图级结果通过（最新一次原始样例结果中，期望方案严格字符串为 4/5）





## 后续进度（2026-09-13）

- 浏览器聊天页面：完成
- FastAPI 根路径静态页面：完成
- 浏览器端 SSE 增量渲染与 session_id 保存：完成
- 页面运行级验证：`GET /` 返回 200，页面标题和聊天接口引用存在

## 后续进度（2026-09-13）

- 浏览器对话模式：完成
- 浏览器售后提取模式：完成
- 售后提取结果卡片：完成
- 根路径页面运行级验证：通过
- 浏览器脚本语法验证：通过

## 质量回归进度（2026-09-13）

- 客服对话评估脚本：完成
- 评估样例：3 组多轮/行为约束样例
- 真实 GLM 评估：待运行 `scripts/eval_chat.py`

## 客服质量回归进度（2026-09-13）

- 客服对话评估脚本：完成
- 真实 GLM 多轮评估：`3/3` cases passed，`6/6` assertions passed
- 覆盖：上下文承接、订单信息追问、禁止虚假退款/换货、未知定制商品政策不编造

## 演示入口（2026-09-13）

- PowerShell 对话脚本：`D:\实习\Project01\scripts\demo_chat.ps1`
- PowerShell 售后脚本：`D:\实习\Project01\scripts\demo_after_sales.ps1`
- 真实 GLM SSE 脚本验证：通过

## 浏览器会话体验（2026-09-13）

- 本地聊天展示记录恢复：完成
- 新建会话清理展示记录：完成
- 服务端会话仍为内存会话，浏览器 localStorage 不作为服务端事实来源

## 安全质量回归（2026-09-13）

- Prompt 注入安全样例：通过
- 虚假退款/换货执行约束：通过
- 未核实物流状态约束：通过
- 真实 GLM 回归：`5/5` cases passed，`9/9` assertions passed
- 确定性测试：`18 passed`

## 一键验证进度（2026-09-13）

- 全自动验证入口：完成
- 编译检查：通过
- Python 自动化测试：`18 passed`
- 浏览器 JavaScript 语法：通过
- 真实 GLM API 冒烟：健康页、浏览器页面、第一轮 SSE、第二轮上下文、结构化提取全部通过

## Chapter 02：Function Calling 与数据库（2026-09-15）

- 设计：完成。
- Docker Desktop/WSL2：已安装并运行。
- MySQL Compose 与四表 DDL：完成；MySQL 8.4 容器与 5 条 FAQ 数据已验证。
- SQLAlchemy ORM、异步仓储：完成并通过 SQLite 隔离测试。
- 五个业务工具：完成；FAQ 数据访问与随机订单/商品/物流工具已实现。
- 工具执行器：完成参数校验、超时、重试和错误包装测试。
- 聊天入口单轮 Function Calling：完成；DeepSeek `deepseek-flash` 真实调用已验证。
- 工具轨迹持久化：完成；MySQL messages/tickets 落库已验证。
- 页面工具徽章：完成；三条验收问法已用内置浏览器可视化验证。
- 四表只读面板：完成；`GET /api/v1/database/tables` 展示最近记录，页面不提供写操作。
- 当前确定性测试：`28 passed`。
- 一键验证：`ALL_CHECKS_PASSED`。

## 浏览器端到端验收（2026-09-13）

- Edge 控制插件：当前 Codex API-key 模式仍返回 `unsupported Codex auth method: apikey`，未伪造 Edge 已通过。
- 浏览器运行时备用验收：通过 Codex In-app Browser 完成实际页面交互。
- 第一轮对话 SSE：通过
- 第二轮上下文承接：通过
- 售后提取页面：通过
- 页面刷新后本地聊天记录恢复：通过
- “新建会话”删除本地记录：未点击，需按安全规则在删除前单独确认

## 浏览器会话失效处理（2026-09-13）

- 失效 session_id 自动恢复：完成
- 自动清理失效本地展示记录：完成
- 当前消息单次自动重试：完成
- 内置浏览器可视化验收：通过
- 自动化测试：`18 passed`
