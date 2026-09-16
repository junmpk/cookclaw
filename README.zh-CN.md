# CookClaw

CookClaw 是一个面向项目展示的 AI 烹饪助手后端，基于 Python 和 FastAPI 构建。项目展示了一个偏生产实践的 Agent Runtime：多通道请求规范化、确定性的领域处理器、混合食谱检索、可持久化的会话状态、受控规划，以及安全的设备操作流程。

> 公开版本说明：私有基础设施、凭据、登录会话、专有数据集和真实硬件厂商适配器均已移除。IoT 边界由本地 Mock 设备表示，不会连接真实硬件。

## 项目设计

核心原则是将语言模型的推理与事实、授权和副作用隔离：

```text
Web / IM 适配器
    -> TurnApplicationService
    -> 有序的 TurnOrchestrator 处理器
    -> 受控 Planner（可选，仅允许低风险动作）
    -> RAG / 记忆 / Mock 设备端口
    -> ResponseEnvelope
    -> 通道渲染器
```

- LLM 负责理解意图并组织自然语言响应。
- 食谱事实必须来自已配置的检索存储。
- 设备操作需要明确确认，并经过确定性的状态检查。
- 副作用超时会记录为未知状态，不会被盲目重试。
- 工具证据和状态变更与面向用户的文本分离。

## 主要能力

- FastAPI + SSE Web API，以及共享的 QQ、微信和 WhatsApp 请求门面。
- Milvus 混合检索：稠密向量 + BM25 + RRF + 重排。
- 搜索澄清、候选选择、结果比较和菜单规划。
- Redis 短期任务状态与 PostgreSQL 长期用户资料边界。
- 可选的受控 Planner，支持关闭、影子运行和白名单主动模式。
- 面向 Trace / Replay 的失败数据集与回归测试。
- 本地 Mock 设备适配器，用于安全演示确认和幂等流程。

## 目录结构

```text
app/
  api/                    FastAPI 路由
  orchestrator/           回合运行时、路由和领域处理器
  conversation/           Redis/PostgreSQL 状态与记忆适配器
  agent/                  模型、检索和受控 Agent 集成
  qqbot|weixinbot|whatsapp 通道适配器
tests/                    单元测试与回归测试
scripts/                  本地灌库、评测和维护脚本
db/                       数据库迁移与校验 SQL
docs/                     架构和行为规范
```

## 快速开始

环境要求：Python 3.12+、[uv](https://docs.astral.sh/uv/) 和 DashScope API Key。复制公开配置模板，并将生成的 `.env` 保留在本地：

```bash
cp .env.example .env
uv sync
uv run python -m app.main
```

然后检查本地健康接口：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

食谱检索需要你自行准备并获授权的数据集，以及 Milvus Lite 或 Milvus Server。本仓库不分发食谱工作簿或生产数据库。

### 公开 Web 演示

公开副本还包含一个 Web 演示端：左侧聊天复用 QQ/WhatsApp 的共享对话入口。
普通消息先进入共享对话 Runtime；系统从菜单、饮食、库存和调度四个维度判断任务复杂度，
复杂配餐会自动桥接到 LangGraph 动态多 Agent 流程，并按需启用 3～5 个角色。
普通聊天、图片识别确认和多 Agent 共用同一个 `thread_id` 与真实混合 RAG；设备执行始终停留在 Mock 边界：

```bash
uv run python -m app.demo.seed
uv run python scripts/verify_interview_demo.py  # 离线演示契约自检
uv run uvicorn app.demo.server:app --host 127.0.0.1 --port 8010
```

打开 <http://127.0.0.1:8010>。页面最左侧是 WhatsApp 风格的厨房聊天窗口：
闲聊、厨房技巧、补充约束、候选替换和菜谱详情都在同一条共享会话中完成；也可
上传 JPEG、PNG 或 WebP 冰箱照片，先核对视觉模型识别的食材，再补充食材或开始检索。
复杂配餐自动路由后可以观察动态组队原因、并行时间线、共享状态、库存覆盖与烹饪排期，并在同一聊天框
回复“确认”或“取消”，也可以在确认前说“第二道换掉”触发局部修订；中间的
食谱卡片会展示来源图片，远程图片暂时不可用时仍会显示可识别的占位区域，
不会阻塞流程。调度 Agent 只在限时、并行或有限设备等条件下加入；公开数据缺少生产
步骤时，页面会把虚拟步骤明确标为“演示模板”，不把它当作真实做法或设备指令。
右侧指标来自实际事件账本，展示模型/工具调用、节点数、耗时和 checkpoint 恢复次数；
`thread_id → run_id → checkpoint` 的身份也会随 Run 持久化。
真实模型调用从本地 `.env` 读取 `DASHSCOPE_API_KEY`；流程演练
模式不调用模型。

`DEMO_RECIPE_BACKEND=auto` 会在 `RECIPE_MILVUS_URI` 可用时选择主工程的
稠密向量 + BM25 + RRF + 重排链路；公开克隆没有授权数据时，运行
`python -m app.demo.seed` 后会明确降级为 `public_rehearsal`。页面顶栏会显示当前
后端，规则演练库不会伪装成真实混合 RAG。六项第一轮改造的链路、状态关系和
面试演示脚本见 [Web 完整闭环](docs/web-agent-workflow.md)。

如需在面试演示中让 QQ、微信和 WhatsApp 也进入同一张 LangGraph，可按
[第三阶段 IM Graph bridge](docs/im-graph-phase3.md) 打开默认关闭的灰度开关。
通道内支持“第二道换掉”“把第二道换成西兰花烤土豆”“第一道清淡一点”等
局部调整；系统只改变指定槽位，并重新执行校验与确认。

## 配置注意事项

公开配置模板位于 `.env.example`。请遵守以下规则：

- 永远不要提交 `.env`、通道登录状态、数据库文件或设备凭据；
- 使用 `RECIPE_MILVUS_URI`，不要使用 pymilvus 保留的 `MILVUS_URI`；
- 在配置真实凭据前，保持所有真实通道集成处于关闭状态；
- 仓库内置的设备适配器仅用于 Mock，不能控制真实硬件。

## 验证

```bash
uv run python -m compileall -q app
uv run pytest -q
git diff --check
```

更多信息请阅读[架构文档](docs/architecture.md)、[Agent 规范](docs/agent-spec/README.md)和[公开发布边界](OPEN_SOURCE_BOUNDARY.md)。

## 项目状态

本仓库是技术展示项目，不是托管服务，也不是生产级 IoT SDK。真实账号通道验收、生产基础设施、专有食谱数据和硬件厂商认证均不属于公开仓库范围。
