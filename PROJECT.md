# 电商智能客服系统：Chapter 01

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

## 进度

- 设计：完成
- 基础配置和模型客户端：完成
- 对话 SSE：完成
- 内存会话与上下文裁剪：完成
- 售后结构化提取：完成
- 浏览器聊天页面 / 售后提取页面：完成
- 确定性自动化测试、健康检查和真实上游验收：完成