# 07. Memory Specification

## 1. 当前记忆分层（AS-IS）

| 层 | 载体 | 主要内容 | 默认时效 |
|---|---|---|---|
| 单轮任务工作区 | `TurnScopedDialogueStatePort`（内部随机 key，复用 `task_state_workspace.py` 状态语义） | 当前一轮同步业务代码访问的候选、选择、排除、pending、active | 仅当前 Turn；退出即清理，不是事实源 |
| 独立任务状态 | Redis `task-state:<thread_hash>` | `ConversationTaskState` + 独立 `revision` + `generation` | 默认 24 小时；文本、Web 稳定 session 与图片链共用 |
| 对话记忆 | Redis ConversationService | 最近轮次、摘要、偏好、搜索历史、食物事件 | 默认 24 小时；与任务状态冲突域分离 |
| 单轮运行时快照 | `RuntimeMemorySnapshot` → `_QQTurnRuntime.memory_snapshot` | Redis 短期会话隔离副本 + PostgreSQL 长期画像当前有效投影 + 分层加载状态 | 仅当前业务轮次，不持久化 |
| 账号画像事实源 | PostgreSQL `channel_users` | 已确认称呼、跨会话饮食偏好、规范 digest、食物事件、职业/家庭/习惯/目标等通用事实 | 长期保存；阶段事实可过期 |
| 通用画像派生索引 | Milvus `cookclaw_user_memory_v1` | 匿名画像哈希、事实 ID、规范文本和向量 | 可删除、可重建，不是事实源 |

实现主要位于：

`app/conversation/` 是记忆、会话和任务态基础设施，不是编排层。单轮执行顺序与
Handler 调度由 `app/orchestrator/turn/` 负责；Conversation 模块只按 Port/Repository
契约读取、投影和提交状态，不选择意图、工具或回复。

- `app/conversation/service.py`
- `app/conversation/runtime_memory.py`
- `app/conversation/store.py`
- `app/conversation/task_state_store.py`
- `app/conversation/task_state_repository.py`
- `app/conversation/task_state_workspace.py`
- `app/conversation/models.py`
- `app/conversation/profile_fact_extractor.py`
- `app/conversation/profile_facts.py`
- `app/conversation/general_profile_command.py`
- `app/conversation/milvus_memory_index.py`
- `app/ports/dialogue_state.py`
- `app/ports/task_state.py`
- `app/agent/participle_agent.py`

## 2. 当前记住什么

- 最近用户和助手轮次；
- 压缩后的对话摘要；
- 抽取出的偏好、忌口或纠正；
- 最近一次真实搜索和历史搜索快照；
- 搜过、选过、执行过的食物事件；
- 当前候选组及更大的候选池；
- 明确选中的菜谱 ID、排除菜谱 ID（候选原序号不改变）；
- 设备确认、会话焦点和活动烹饪任务。
- 用户明确设置并已成功写入 PostgreSQL 的称呼。
- 用户在原文中明确自述、通过受控 Extractor 校验并成功写入 PostgreSQL 的职业、行业、工作方向、家庭、习惯、目标、厨艺画像、沟通偏好和兴趣。

默认保留最多约 20 条用户轮次；上下文拼接偏向最近 6 轮，并受约 6000 token 预算限制。

## 3. 什么时候读

1. 新 Turn 每轮由 `RedisDialogueTaskStateRepository` 读取一次独立 Redis task-state record，并把完整基线装入随机内部 key 的单轮工作区；真实 `thread_id` 不进入 `task_state_workspace.py` 的进程字典；
2. 显式称呼、偏好、纠正、清除等 Memory 命令先执行，防止本轮继续使用修改前的画像；
3. Memory 命令未终止本轮时，`ConversationService.load_runtime_memory()` 一次读取短期会话和账号长期画像，构造 `_QQTurnRuntime.memory_snapshot`；新 thread 没有 Redis 记录时仍加载同一账号的 PostgreSQL 画像；
4. Router、Planner、问候、推荐与 QA 从同一个快照派生各自的裁剪视图，不在同一轮重复读取画像；
5. 读取候选、选择、排除和焦点，判断“这三道/第二个/换一批/上一批”；
6. “你还记得我是谁吗/你了解我什么”等资料查询使用快照中的全部当前有效通用事实；
7. 普通闲聊/问答按当前问题从 Milvus 做用户内语义召回，再按快照中的 PostgreSQL 当前有效事实求交；索引异常时降级读取最近 PostgreSQL 事实；
8. 只有泛化搜索时，才读取饮食画像作为弱偏好；通用职业/兴趣等事实不进入 `SearchRequest` 或 Planner payload；
9. 设备确认、取消、停止和进度只读取对应确定性状态机，通用画像不能授权动作；
10. Web 由服务端签发 128-bit 高熵 `X-CookClaw-Thread-Id`；客户端只有持续复用该 bearer 才能延续 transcript 与任务状态。匿名 Web 不获得跨会话账号画像。

### 3.1 单轮水合契约

- `short_term_status` 区分 `loaded / new / unavailable / invalid / unsupported`；只有 Redis 确认不存在记录时，`is_new_session` 才是 `true`，读取失败或 payload 身份与 key 不一致时必须是未知；
- `task_state_status`、`task_state_revision`、`task_state_generation` 独立于聊天加载状态；独立 key 尚不存在且旧 conversation 读取失败时必须 fail closed，不能把未知旧状态当成空状态；
- `long_term.status` 区分 `loaded / empty / unavailable / invalid / unsupported`，不能把 PostgreSQL 故障或损坏资料解释为“用户没有长期资料”；
- 文本或图片菜谱搜索/推荐遇到长期画像 `unavailable / invalid` 时，不能假定历史过敏与忌口为空；必须让用户明确本次饮食安全事实后才检索；画像可读时图片链也必须合并长期过敏/忌口并做结果硬过滤；资料查询则直接说明暂时读不到；
- 水合是纯读取：新 thread 不因此创建 Redis key，长期画像也不会复制进 Redis；PostgreSQL 仍是长期事实唯一来源，避免删除后的旧资料被短期副本复活；
- PostgreSQL 的清除时间与偏好删除/过期墓碑会屏蔽旧 thread 中的陈旧值；清除后只有用户本轮重新明确表达的偏好可作为临时覆盖，随后仍须按正常写链提交画像；
- 每个快照携带 `profile_version` 和 `hydrated_at`，一轮内保持一致；下一轮重新读取，以看到刚完成的修改或清除；
- “新对话”和“清除全部记忆”是 UoW 内首个确定性 handler：任务态与 conversation 都写入递增 `generation` 的空墓碑；旧 worker 不得跨代合并或复活清理前状态；
- 撤销、删除、过期以及 TTL 损坏的阶段事实不进入快照；通用事实只通过受控 QA 视图读取；
- Trace 只记录加载状态、版本和计数，不记录 thread、用户 ID、称呼、偏好或事实正文；运行时对象的默认 `repr` 同样隐藏这些值；
- 快照不能写回任何存储，也不能作为设备选择、启动、停止或确认的授权依据。

### 3.2 第一阶段类型化消费视图

- `SafetyMemoryView`、`PlannerMemoryView`、`QAMemoryView` 是冻结的消费边界；嵌套值只使用标量、元组和冻结记录，不向调用方暴露 `ConversationMemory`、字典或列表引用；
- 原始视图保留长期、会话和当前轮来源，便于审计，但不能自行合并。生产消费者只能使用 `ConversationService` 设置 `policy_resolved=true` 后的 `effective_*` 字段；
- 合并策略仍唯一位于 `ConversationService`：先处理 PG 删除/过期墓碑和 `food_memory_cleared_at`，再按既有优先级合并长期与会话偏好，最后应用当前轮纠正并拆分阶段约束；
- Planner 和 QA 直接消费类型化视图；Router 的 `preferences / route / full` 字符串 scope 暂时保留为兼容门面，但其关键偏好字段同样来自已解析视图；
- `loaded/empty` 的空值是权威结果，不能因列表或字典为空而回退到旧 `RoutingContext`；`unavailable/invalid` 也不能复活可能陈旧的值；
- 兼容字典每次重新生成深层副本，修改一个消费者拿到的结果不会污染同轮快照或其他消费者。

## 4. 什么时候写

新 Turn 的普通任务状态写入遵循统一事务边界：

```text
load independent task record once
→ install turn-local DialogueStatePort
→ handler 修改工作区
→ 成功时 strict CAS commit
→ revision 冲突时只三方合并非重叠字段
→ generation 变化或同字段分叉时拒绝提交
→ handler 异常时丢弃工作区
```

设备 handler 在真实副作用完成后立即 flush，不等整轮响应退出；Redis 原子 claim 返回
领取后的完整权威状态、revision 和 generation，失败 worker 不得复活 pending。

- 用户和助手完成一轮有效对话后写 turn；
- “新对话/清除全部记忆”先等待该 thread 在命令前已入队的后台 turn/profile 写链结束，再落重置或清除标记；旧队列不得跨过清除边界复活资料；
- 搜索成功后写最近搜索、搜索历史、候选快照和食物事件；
- 用户明确选择、排除或反悔时，只更新结构化 recipe ID，不改写候选顺序；
- 用户明确表达长期偏好/忌口时写 profile；
- “请叫我 X”由确定性解析器同步写 `preferred_name`；已有不同称呼时先把修改请求写入 Redis，用户确认后再提交；
- 明确“请记住/忘掉我的职业、习惯、目标等”先经一次无工具结构化 Extractor 和确定性校验，再同步写 PostgreSQL；只有 PG 成功后才生成自然确认；
- 普通自述在助手回复完成后被动提取，不影响当前轮路由或回复；
- 通用事实 PG 写入成功后尽力同步 Milvus；索引失败不回滚事实源，后续可用 `scripts/reindex_user_memory.py` 重建；
- 用户纠正记忆时撤销或覆盖对应事实；
- 设备动作进入确认、执行、停止时更新状态机；
- 工具失败不应被写成“用户偏好”或“成功行为”。

密码、密钥、验证码、联系方式、精确地址和服务器连接信息在调用 Extractor 前即拒绝；
候选还必须命中字段白名单、只含正常敏感级别，并能在本轮用户原文中找到证据。

图片搜索产生的候选和待补充食材动作已经写入 `task_state`；当前仍未把图片搜索
完整写入 `food_history`，因此它可以支撑重启后的序号追问，但不会被当作长期饮食行为。

## 5. Thread 与账号隔离

当前 IM 线程通常按通道和用户构造：

- QQ 私聊：账号用户；群聊还包含群和发送者；
- WhatsApp：私聊用户；群聊区分成员；
- 微信：bot account + user；
- 账号画像：`profile:{channel}:{user_id}`。

同一通道账号下的不同 thread 可以复用同一个 profile；不同通道仍是不同身份。当前没有
canonical account 映射，因此不能因为 QQ 与微信显示名相同就自动合并记忆。

Web `/api/v1/chat` 在缺少 ID 时生成唯一 `web:<32hex>` thread，并通过
`X-CookClaw-Thread-Id` 响应头返回；后续请求可在 body 中复用。HTTP 边界只接受
同格式高熵 token，低熵自定义值和 IM thread 会返回 422。Web 输入会被收进
`web:` 命名空间；该稳定 session 可恢复近期 transcript、摘要、候选、焦点和 pending。
匿名 Web 仍没有跨设备稳定账号；登录用户体系
上线后还需映射 canonical account id。

## 6. 清除与纠正

### 6.1 清除

- 删除当前 thread 的近期对话、摘要、偏好和搜索状态；
- 用户明确要求时同步删除账号画像中的对应信息；
- “清除记忆/删除我的所有资料”会物理移除称呼、饮食画像和通用事实值，并尽力删除对应 Milvus 派生记录；
- 通用事实的单项忘记会先改 PostgreSQL，再删除对应 `fact_id` 的派生索引；索引删除失败不改变已经完成的事实源删除；
- 写入最小化的“已清除”标记仅用于防止本轮回填；
- 清除记忆不等同于停止设备，必须单独确认设备动作。

### 6.2 纠正

“我没说要低脂”应删除错误事实或标记失效，而不是追加一条相反文本。后续检索不得再使用被撤销事实，并应简短确认：已更正，接下来按当前要求判断。

## 7. 当前缺陷

| 缺陷 | 影响 |
|---|---|
| Web 客户端未复用返回的 thread | 每次请求会生成新会话，无法延续候选/焦点 |
| `task_state_workspace.py` 仍保留同步状态语义 | 统一链只通过 Port 用随机 key 复用；它不是跨轮 store，也不拥有编排职责，后续可继续迁成纯 reducer |
| TaskState 默认 TTL 仍为 24 小时 | active 状态过期后必须查询真实设备；Redis 不能替代设备事实源 |
| 图片搜索未进入长期 food history | 重启后能恢复候选，但不能用于长期行为统计 |
| 测试服务器的 Milvus Server/RBAC 尚未验证 | 代码预检已实现，但真实建集合、upsert、delete 和召回仍是待确认 |
| 没有跨通道 canonical account | QQ、微信、WhatsApp 的同一自然人不会共享画像 |
| “我喜欢 X”存在饮食偏好/普通兴趣歧义 | 当前优先确定性饮食偏好，部分自然兴趣表达需用真实语料继续消歧 |
| Milvus 是异步尽力同步 | 短时故障可能造成索引漂移，但 PG 事实不丢；需监控并按需重建 |
| 没有面向用户的画像查看/导出入口 | 当前只能通过对话查询、纠正和清除，商业化前需补产品与审计能力 |

## 8. 通用画像事实数据模型（AS-IS）

每条可复用事实至少包含：

~~~json
{
  "id": "mem_<profile-and-value-hash>",
  "type": "profile_fact",
  "category": "work",
  "key": "occupation",
  "value": "AI应用开发工程师",
  "subject": "self",
  "scope": "stable",
  "source": "user_explicit",
  "source_channel": "qq",
  "source_thread_hash": "<sha256-prefix>",
  "confidence": 1.0,
  "status": "active",
  "created_at": 0.0,
  "updated_at": 0.0
}
~~~

该结构保存在 PostgreSQL `channel_users.long_term_memory.facts`。稳定单值槽位（例如职业）
覆盖旧值；兴趣、家庭成员等允许多值；阶段事实带 `expires_at`。删除和过期值会物理移除，
Milvus 只保存可重建的规范文本，不保存 `value` 之外的原始证据句或 thread 身份。

### 8.1 Extractor 权限边界

- 单次最多提出 3 条候选，严格 Pydantic schema，禁止额外字段；
- 只允许受控 category/key 组合；
- LLM 不能调用工具、不能读现有画像、不能直接执行写删；
- 称呼、饮食偏好、过敏和设备状态由已有确定性模块独占；
- 资料查询只读取当前有效事实，已删、过期或仅存在于 Milvus 的记录不能回答给用户。

### 8.2 派生索引一致性

```text
写/改/删：PostgreSQL commit → best-effort Milvus 同步
普通召回：Milvus 用户过滤 → score 门槛 → fact_id 与 PostgreSQL 求交
索引故障：PostgreSQL 最近事实降级
全量恢复：PostgreSQL → scripts/reindex_user_memory.py → Milvus
```

当前 `ConversationTaskState` 位于 `app/conversation/models.py`，主要字段及职责：

| 字段 | 写入者 | 读取者 | 生命周期/清理 |
|---|---|---|---|
| `candidate_recipes` / `candidate_decision_id` | 真实检索提交、图片检索 | 序号/指代解析、路由摘要 | 新搜索覆盖；候选 TTL 或清空会话后移除 |
| `active_search_request` | 首次真实检索、条件增量 reducer | 条件追加/修改、失败重试、路由裁剪摘要 | 候选失效时保留；清空会话或 6 小时活动 TTL 后移除 |
| `active_search_updated_at` | active search reducer | 语义 TTL 投影 | 只在 active search 本身改变时推进；无关 task patch 不续期 |
| `selected_recipe_*` | 用户明确选择、设备预检 | “这个/刚才那个”、设备流程 | 新候选、明确排除或清空会话时移除 |
| `excluded_recipe_ids` | 用户明确排除/反悔 | 后续选择和回复 | 不删除候选，避免序号漂移；新候选时清空 |
| `pending_search_clarification` | 搜索澄清编排 | 下一轮 SearchRequest 合并 | 搜索完成、取消或超时清除 |
| `pending_profile_update` | 称呼命令状态机 | 下一轮确认/取消 | 确认、取消或 5 分钟后清除；不得和设备确认共用 |
| `temporary_profile` | PostgreSQL 写入失败降级 | ContextBuilder | 仅当前 Redis 会话；不能对用户声称已长期保存 |
| `pending_action` | Orchestrator | 下一轮短回复解释 | 动作消费、取消或超时清除 |
| `pending_device_start` | 设备预检 | Redis 原子确认领取 | 领取时原子迁移为 `device_execution`；取消或 5 分钟 TTL 后清除 |
| `device_execution` | Redis 原子领取、设备 Handler | 重复确认门禁、未知结果恢复、审计 | `dispatching/submitted_unverified/outcome_unknown` 保留；明确失败可重试，停止成功后清除 |
| `active_cooking` | 设备已验证启动结果 | 停止、进度、路由 | 关联 `action_id/msg_id`；停止成功、明确清理或会话 TTL 到期 |

任务态保存在独立 Redis key；`ConversationMemory.task_state` 只是迁移期兼容镜像，
`task_state_storage_version=2` 后即使独立 key 过期也不得从镜像复活。同一轮结束时按独立 task revision 严格 CAS。发生并发时只允许三方合并非重叠字段；
同字段分叉或 reset generation 变化必须失败。Redis 使用 Lua 将
`pending_device_start` 原子迁移为带 `action_id/msg_id` 的 `device_execution(dispatching)`，
并返回领取后的完整状态；领取失败或存在未决执行时不得再次下发设备指令。
会话清理用单个 Lua 同时推进 conversation/task generation，旧 turn 不得跨代回写。
普通闲聊和只读设备查询不清理菜谱任务；显式“继续说刚才的菜”只恢复原候选，
不重新检索也不改变原序号。条件发生变化时旧候选立即失效，但
`active_search_request` 保留到会话清理或活动 TTL 到期。

## 9. 使用优先级

~~~text
本轮明确纠正/排除
  > 本轮明确要求
  > 明确过敏与长期安全约束
  > 近期对话中仍有效的上下文
  > 长期偏好
  > 搜过/点过/做过的弱行为信号
~~~

历史行为永远不能替用户做主。回复中只有当历史确实改变推荐时才自然解释一次；不要每轮机械声明“历史只当参考”。

通用职业、家庭、兴趣和沟通偏好只用于自然回复。它们不得自动转成菜谱硬过滤、设备选择、
设备授权或执行确认；如果用户本轮提出菜谱/设备要求，仍以当前原话和确定性状态为准。

## 10. 称呼记忆规则

- PostgreSQL `channel_users.preferred_name` 是已确认称呼的唯一长期事实源；
- `long_term_memory.facts` 中的称呼记录只保留来源和变更审计，不作为当前称呼的第二事实源；同一字段中的 `profile_fact` 则是职业、家庭等通用事实的当前事实源；
- Redis `pending_profile_update` 只保存尚未确认的修改；
- PostgreSQL 写入失败时可用 `temporary_profile` 保持当前会话体验，但回复必须明确长期保存未成功；
- “嗯/确认”必须结合当前待确认状态解释。资料修改和设备启动同时待确认时必须追问具体对象，绝不能同时消费；
- 昵称作为不可信用户数据处理，长度最多 24 字，并拒绝 URL、代码和明显提示注入内容；
- Deep Agent 只能读取已确认称呼，不得直接写用户画像。

## 11. 隐私与保留

- 只为产品目标保存最少数据；
- 内部测试日志分析必须限定授权测试账号，先脱敏再聚合；
- 原始手机号、群号、消息 ID、图片和设备 ID不得进入日报或 Prompt 评测集；
- 用户应能查看、更正、清除长期偏好；
- 生产环境应有明确保留周期、访问控制和删除审计；
- 不允许把聊天日志直接用于自动改 Prompt 或自动上线规则。

## 12. 开启前预检与回滚

`CONVERSATION_GENERAL_MEMORY_ENABLED` 默认是 `false`。启用后，部署初始化脚本
`scripts/init_data_stores.py` 会在服务重启前执行以下检查：

1. `RECIPE_MILVUS_URI` 必须指向 Milvus Server；可选的 `MEMORY_MILVUS_URI` 也必须是 Server URI；
2. `MEMORY_MILVUS_COLLECTION` 必须是独立集合，不能与菜谱集合同名；
3. 目标集合不存在时实际创建；目标集合已存在时创建并立即清理一个随机空集合，重新验证当前账号的 create 权限；
4. 校验目标集合 schema 和向量维度并 load，再使用不含用户数据的匿名探针实际执行 delete → upsert → delete；任一步失败即终止初始化，不进入服务重启。

清理随机空集合需要 `DropCollection` 权限；这是为了不在每次预检后留下垃圾集合，不会删除
目标集合或任何用户资料。

紧急回滚只需把总开关恢复为 `false` 并重启。PostgreSQL 已保存事实不会因此丢失；关闭后
不再提取、召回或写入通用画像，原称呼与饮食偏好链保持不变。
