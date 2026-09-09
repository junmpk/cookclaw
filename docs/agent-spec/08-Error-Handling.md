# 08. Error Handling Specification

## 1. 目标

错误处理的第一目标不是“说得好听”，而是让用户知道：发生了什么、动作是否真的完成、现在能做什么。任何不确定状态都不得包装成成功。

## 2. 当前主要降级路径（AS-IS）

- 搜索：进程内常驻服务失败 → 子进程回退 → 两者失败后返回搜索异常；
- Rerank：可退回融合召回顺序；
- 意图分类：解析失败通常变为 unknown，再进入 QA/闲聊路径；
- 图片：下载、识别或上传失败时，各通道分别拼接错误文案；
- 设备启动：先检查状态，发命令后轮询运行状态；
- 通道发送：部分通道对媒体失败有文本兜底；
- 顶层异常：主入口会记录日志，但部分 IM 路径可能没有向用户发送最终失败回复。

当前问题是异常文案、重试和日志分散在多个 adapter/orchestrator 中，缺少统一错误码和 trace id。

## 3. 统一错误码（TO-BE）

| 错误码 | 含义 | 是否可重试 | 用户策略 |
|---|---|---:|---|
| `INTENT_PARSE_FAILED` | 无法可靠识别意图 | 是 | 不暴露解析错误；按上下文保守回答或问一个问题 |
| `SEARCH_UNAVAILABLE` | 搜索依赖不可用 | 是 | 说明暂时搜不了，不给编造候选 |
| `SEARCH_TIMEOUT` | 搜索超时 | 是 | 保留条件并允许原话重试 |
| `SEARCH_EMPTY` | 安全过滤后无结果 | 否 | 建议只放宽一个非安全条件 |
| `VISION_UNCERTAIN` | 图片识别置信不足 | 否 | 说明不确定并追问一个关键点 |
| `VISION_FAILED` | 图片处理失败 | 是 | 请重发清晰图片或直接描述食材 |
| `DEVICE_CONFIG_MISSING` | 无可用设备配置 | 否 | 引导绑定/配置，不发送命令 |
| `DEVICE_OFFLINE` | 设备离线 | 是 | 请检查联网，稍后重试 |
| `DEVICE_BUSY` | 设备正忙 | 否 | 告知当前状态，不覆盖任务 |
| `DEVICE_STATUS_TIMEOUT` | 状态查询超时 | 是 | 明确状态未知，不说已启动/已停止 |
| `DEVICE_COMMAND_REJECTED` | 设备拒绝命令 | 视原因 | 展示可操作原因，不重复轰炸 |
| `DEVICE_STATE_UNVERIFIED` | 命令返回但状态未确认 | 是 | 明确“已发送，尚未确认执行” |
| `CHANNEL_SEND_FAILED` | 通道消息发送失败 | 是 | 服务端补偿/告警，避免重复设备动作 |
| `MEMORY_DEGRADED` | 记忆服务不可用 | 是 | 继续当前轮，但不假装记得历史 |
| `UNAUTHORIZED` | 用户/回调/设备未授权 | 否 | 拒绝动作并给合法入口 |

## 4. 用户回复结构

推荐结构为两句，必要时三句：

1. 当前事实：哪里失败或状态未知；
2. 影响边界：搜索没完成、设备未确认、图片没识别清；
3. 一个下一步：重试、检查联网、换图片或放宽条件。

禁止：

- 直接输出堆栈、环境变量、内部 URL 或密钥；
- 笼统说“系统繁忙”却不说明动作是否执行；
- 在设备状态未知时说“已经开始/已经停止”；
- 搜索失败后现场编三道菜；
- 无限重试写操作。

## 5. 重试与幂等

| 动作 | 自动重试 |
|---|---|
| Embedding/Rerank/只读状态 | 可有限次指数退避 |
| 搜索进程内失败 | 可切换到子进程一次 |
| 通道文本发送 | 可按 message id 幂等重试 |
| 图片上传 | 可重试；最终以文本兜底 |
| 设备启动/停止 | 不得盲重试；必须使用幂等键并先查状态 |
| 记忆写入 | 可重试；失败不得阻塞核心只读回答 |

## 6. 健康检查

当前应用 health 更接近“进程活着”，不能证明 WhatsApp、Milvus、DashScope、数据库或设备 API 可用。目标应拆为：

- **Liveness**：FastAPI 进程和事件循环是否存活；
- **Readiness**：检索、必要数据库和启用通道是否已就绪；
- **Dependency status**：Milvus、Embedding、Rerank、QQ/微信/WhatsApp、设备 API 分项状态及最近成功时间；
- **Degraded**：可降级依赖失败时明确标记，而不是整体 200 `ok`。

## 7. 日志与可观测性

每次请求 SHOULD 贯穿一个 trace id，并记录：通道、匿名用户键、意图、上下文动作、工具、耗时、降级、错误码和最终发送状态。

MUST 脱敏：手机号、QQ/微信标识、群号、消息 ID、设备 ID、图片 URL token、原始认证头。原始聊天内容不应默认写 INFO 日志；内测分析需单独授权和隔离存储。

核心指标包括：

- 意图失败率和候选追问误路由率；
- 搜索成功率、空结果率、降级率、P95；
- 设备命令接受率、状态确认率、误操作/重复操作数；
- 通道发送成功率与媒体降级率；
- 无用户最终回复的请求数，目标为 0。

### 7.1 当前已落地的对话 Trace

Web、QQ、微信、WhatsApp 文本及统一图片入口会创建 `conversation_trace_v1`，同一异步
请求中的 Intent、Router、Planner、受控 Agent、检索、视觉和设备 Tool 共用一个
`trace_id`。实现位于：

- `app/observability/trace.py`：ContextVar 隔离、计数、Token、耗时和 JSONL；
- `app/observability/baseline.py`：P50/P95 与失败率汇总；
- `scripts/summarize_conversation_traces.py`：读取独立 JSONL、systemd 日志或
  `run_qq_business_test.py` 的 JSON 报告。

每轮记录总耗时、模型/意图/Tool/搜索调用次数、最大单次模型输入 Token、
总 Token、搜索耗时、回复模型耗时、超时、Tool 失败、路由结论、降级原因及
最终通道发送耗时。Token 以模型供应商实际返回的 usage 为准；供应商未返回时
保持 0，不能把 0 解释为真实未消耗。

默认只写应用 INFO 日志：

```bash
journalctl -u cookclaw --since today -o cat > /tmp/cookclaw-qq.log
.venv/bin/python scripts/summarize_conversation_traces.py \
  /tmp/cookclaw-qq.log --channel qq
```

也可以让应用写独立 JSONL（父目录必须预先存在且应用用户可写）：

```bash
CONVERSATION_TRACE_JSONL_PATH=outputs/conversation-traces.jsonl
.venv/bin/python scripts/summarize_conversation_traces.py \
  outputs/conversation-traces.jsonl --channel qq --deep-agent false
```

灰度对比时应分别保留 Deep Agent 关闭和开启的样本，使用
`--deep-agent false/true` 分桶。这里的 `deep_agent_enabled` 表示该轮所在
QQ 配置组已开启能力，不等于该轮一定进入了 Deep Agent；实际参与情况还需结合
`deep_agent_used` 或 `deep_agent_model.*` 事件：

```bash
.venv/bin/python scripts/summarize_conversation_traces.py \
  outputs/conversation-traces.jsonl --channel qq --deep-agent-used true
```

### 7.2 对话有效率与路由正确率

运行时 Trace 不能仅凭回复非空判断“对话有效”。本地 QQ 业务回归器继续用每轮
事实断言计算 `effective_turn_rate`；只对声明了 `intent_categories` /
`route_actions` 的轮次计算 `intent_accuracy` / `route_accuracy`，未声明的
轮次不进入对应正确率分母：

```bash
.venv/bin/python scripts/run_qq_business_test.py
.venv/bin/python scripts/summarize_conversation_traces.py \
  outputs/qq_business_tests/qq-business-test-<时间>.json --channel qq
```

真实 QQ 的人工有效性评审、样本量和正式阈值当前仍是**待测基线**。在第一批
真实数据产生前，不预设 P50、Token 或有效率数字。

### 7.3 QQ 定向灰度

受控 Deep Agent 由 `app/agent/deep_agent_rollout.py` 统一判断，顺序固定为：

1. `CONVERSATION_DEEP_AGENT_ENABLED=true`；
2. 通道命中 `CONVERSATION_DEEP_AGENT_CHANNELS`；
3. QQ openid 命中 `CONVERSATION_DEEP_AGENT_QQ_ALLOW_FROM`，或命中
   `CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT` 的稳定哈希分桶。

默认百分比为 0，所以只打开总开关不会误覆盖全部 QQ 用户。定向验证时先只配置
测试账号 openid；确认基线后再显式提高百分比。回滚只需把总开关设回 false，
确定性 Router、设备确认和安全门禁不受该开关影响。

称呼长期记忆使用独立开关和白名单：

```dotenv
CONVERSATION_PROFILE_MEMORY_ENABLED=true
CONVERSATION_PROFILE_MEMORY_CHANNELS=qq
CONVERSATION_PROFILE_MEMORY_QQ_ALLOW_FROM=
CONVERSATION_PROFILE_MEMORY_ROLLOUT_PERCENT=100
```

明确称呼写入必须等待 PostgreSQL 结果。写入超时或失败时仅保存 Redis 会话覆盖，
回复“本次对话先这样称呼，长期保存未成功”；不得回复“已经记住”。称呼修改和
设备启动同时待确认时，裸“嗯/确认”必须追问确认对象，不能下发设备命令。
称呼查询必须直接读取结构化画像；不得从历史任务摘要推断用户身份。若人为关闭
称呼记忆，明确的设置/查询请求必须返回未启用说明，不能回落到自由对话后承诺保存。
Trace 的 `profile_memory` 事件只记录 `action/status/conflict`，不记录称呼原文；
基线汇总输出事件数、动作/状态分布和写入失败率。

### 7.4 核心验收与灰度比较

十组核心多轮场景已收敛为一个不调用真实设备、LLM、Milvus、Redis 或 PostgreSQL
的确定性验收入口：

```bash
.venv/bin/python scripts/run_core_dialogue_acceptance.py
```

它会生成 JSON/Markdown 报告，把原始十组对话、预期状态和对应 pytest 证据关联
起来。真实 QQ 基线产生后，可以比较关闭组与灰度组：

```bash
.venv/bin/python scripts/compare_conversation_baselines.py \
  baseline.jsonl candidate.jsonl --channel qq
```

上面的命令只比较数据，不给出放行结论。真实阈值确认并导出为环境变量后，再运行
门禁命令；以下前置检查会在变量未填写时安全退出：

```bash
: "${MAX_P95_REGRESSION_PCT:?请先填写真实基线阈值}"
: "${MAX_TOKEN_REGRESSION_PCT:?请先填写真实基线阈值}"
: "${MIN_EFFECTIVE_TURN_RATE:?请先填写真实基线阈值}"
: "${MIN_INTENT_ACCURACY:?请先填写真实基线阈值}"
: "${MIN_ROUTE_ACCURACY:?请先填写真实基线阈值}"

.venv/bin/python scripts/compare_conversation_baselines.py \
  baseline.jsonl candidate.jsonl \
  --channel qq \
  --require-deep-agent-used \
  --max-p95-latency-regression-percent "$MAX_P95_REGRESSION_PCT" \
  --max-average-token-regression-percent "$MAX_TOKEN_REGRESSION_PCT" \
  --min-effective-turn-rate "$MIN_EFFECTIVE_TURN_RATE" \
  --min-intent-accuracy "$MIN_INTENT_ACCURACY" \
  --min-route-accuracy "$MIN_ROUTE_ACCURACY"
```

比较器没有内置项目阈值；未传门禁时只输出差异，
`release_decision=not_evaluated`，不能据此宣称灰度通过。

### 7.5 Turn Planner Shadow / Active

Planner 默认关闭。`shadow` 只用于旁路评估：

```dotenv
TURN_PLANNER_MODE=shadow
TURN_PLANNER_MODEL=qwen-plus
TURN_PLANNER_TIMEOUT_SECONDS=8
TURN_PLANNER_SHADOW_MAX_INFLIGHT=4
TURN_PLANNER_SHADOW_JSONL_PATH=outputs/turn-planner-shadow.jsonl
```

Shadow 在 Legacy 路由结束后由后台任务运行，不等待 Planner、不执行 plan、不调用
第二次搜索、不写记忆/状态，也不触发设备。任务使用独立 Context，因此不修改当前
`conversation_trace_v1` 的模型计数或完成时机。进程退出时最多等待 2 秒，之后只取消
未完成的 Planner 模型调用。后台调用达到并发上限时直接跳过本轮 Shadow，不排队
占用主链资源。

Shadow 只覆盖进入对应旁路调用点的轮次；确定性前置命令会记录 bypass，不应把
Shadow 记录数当成全部用户轮次。`active` 则由统一 Turn facade 在设备 pending 之后、
普通 pending/Recipe/Device 之前执行，并受通道、账号/分桶和 action 白名单控制。

日志 schema 为 `turn_planner_shadow_v1`。它只记录上下文计数、Legacy
action/category/risk、Planner phase/action/reply act/risk、Validator 错误码、
机械一致性、耗时与 Token；不记录用户原话、近期轮次文本、偏好值、菜名、菜谱 ID、
step args 或设备标识。

汇总命令：

```bash
.venv/bin/python scripts/summarize_turn_planner_shadow.py \
  outputs/turn-planner-shadow.jsonl --channel qq
```

也可以从 systemd 日志提取。汇总输出只是观测数据，不包含自动放行阈值。以下情况
均 fail closed 且不影响 Legacy 回复：

- 模型超时：`PLANNER_TIMEOUT`；
- 模型调用异常：`PLANNER_MODEL_ERROR`；
- 非 JSON、未知 action 或 schema 越界：`PLANNER_SCHEMA_ERROR`；
- Validator 拒绝：只记录 issue code，不执行任何步骤；
- `active` 越界、高风险或非单步：交回确定性 Handler 或 fail closed；已有工具调用/
  状态变化后禁止再次运行旧链。

### 7.6 生产汇总与告警

`scripts/summarize_conversation_traces.py` 输出 Planner 状态、执行状态、Handler 分布、
fallback reason、工具失败来源和设备安全计数。成功率、P95、超时和 Planner fallback
阈值必须由同环境基线显式传入；发布门禁还应指定最小样本数和设备 unknown 上限。
没有配置阈值时只报告测量值。

以下两项固定零容忍，不依赖业务阈值：

- 同一轮同类设备启动/停止命令发送超过一次；
- Planner execution 出现 `device.start / device.stop / device.confirm / memory.write`。

传入 `--fail-on-alert` 后，有告警时脚本退出码为 `2`，可接入 CI、systemd timer 或
外部监控。

## 8. 各场景回复原则

| 场景 | 回复原则 |
|---|---|
| 搜索不可用 | 保留用户条件，说明暂时无法检索；不得生成菜谱 |
| 搜索 0 结果 | 说清哪类硬条件保留，建议放宽一个软条件 |
| 图片不清楚 | 说明不确定，问菜名/主料中的一个 |
| 设备离线 | 不发启动命令，建议检查联网/电源 |
| 设备启动未确认 | 区分“命令已发出”和“设备已运行” |
| 停止未确认 | 提醒不要假设已停，建议现场确认并重查状态 |
| 记忆不可用 | 当前轮照常，明确这次不会依赖历史 |
| 通道媒体失败 | 用纯文本完整交付结果 |
| 非法 webhook/admin 请求 | 返回 401/403，不进入 Agent 链路 |

## 9. 顶层兜底

所有通道入口 MUST 捕获未分类异常，并尽力发送一次本地化的最终回复；设备写操作则先通过幂等记录查询是否已执行，避免“回复失败”触发重复启动。日志必须保留错误码和 trace id，用户只看到可理解的信息。
