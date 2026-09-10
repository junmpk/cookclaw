# Web 共享对话：第一阶段

## 启动

在公开副本执行 `uv run uvicorn app.demo.server:app --host 127.0.0.1 --port 8010`。
在 `.env` 中配置 `DASHSCOPE_API_KEY`，用 `DEMO_MODEL` 选择模型。
自然语言聊天始终使用模型；高级面板的 rehearsal 仅适用于独立配餐实验。

## 阅读顺序

1. `app/demo/web/app.js`：`sendChatMessage` 只发送消息和会话 ID，不决定业务路由。
2. `app/demo/server.py`：`/api/chat` 校验输入，调用 `handle_web_turn`，返回结构化响应与 Trace。
3. `app/agent/participle_agent.py`：`handle_web_turn` 管理会话锁、用户/助手历史；
   `_run_turn_orchestrator` 与 QQ、WhatsApp 使用同一个回合核心。
4. `app/orchestrator/turn/facade.py` 和 `application_service.py`：任务状态加载、Handler 调度、提交。
5. `app/orchestrator/search_request.py`：把自然语言整理成搜索条件，合并后续修改。
6. `app/orchestrator/menu_plan.py`：菜汤数量校验、单槽位替换。
7. `app/demo/shared_chat.py`：将公开 Milvus 检索结果适配到原食谱协议；保留 ID、来源和原始步骤。

## 演示步骤

- 你好。
- 今晚两个人吃，有鸡肉和西兰花，帮我推荐菜。
- 两菜一汤，不要花生。
- 第二道换掉。
- 第一道怎么做？
- 鸡胸肉怎么做才不柴？

观察学习视图中的 `active_search_request`、`menu_task` 和 `candidate_recipes`。
人数、忌口由后端解析与承接，不依赖高级表单。查看做法使用保存的候选详情。
有可用替代项时换菜只替换一格；无法满足则说明缺少候选。

## 当前边界

- 本地入口明确使用共享 InMemoryConversationStore，进程重启会清空任务状态。
  浏览器保留聊天文字不代表服务器仍有任务；重启后建议新建对话。
- 食谱来自已导入的公开集合，检索是哈希向量基线，不能描述成语义 Embedding。
- 中文菜名为源标题的展示翻译，食材与步骤保留英文原文，不虚构设备程序。
- 普通聊天、搜索和配餐都走共享核心；右侧显示本轮真实 Trace。
- 第二阶段已提供显式“三 Agent 协作” bridge；普通聊天仍走共享核心，详见
  docs/web-chat-phase2.md。
- 不自动启动 QQ/WhatsApp sidecar，不要求登录真实 IM 账号。

## 学习练习

先手写一个 Pydantic 消息模型，再写一个 `async` Handler 返回 `ResponseEnvelope`；
用假的模型或检索端口测试同一会话的连续两轮输入。最后再阅读状态提交和换菜校验。
