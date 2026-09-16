# Web Agent 完整闭环

这份文档对应 GitHub 展示版的第一轮改造。目标不是为了堆 Agent，而是把普通厨房
问答、看图识菜、约束配餐和有副作用的设备动作放进一条可解释、可追踪的业务链。

## 为什么需要动态多 Agent

一次受约束配餐至少包含三类边界不同的工作：

1. **食谱研究 Agent**：只能通过检索工具取得候选，负责查全，不能编造菜名或 ID。
2. **饮食分析 Agent**：读取有来源的饮食规则，负责排除项、未知项和风险提示，
   不在没有食谱证据时推荐菜。
3. **菜单规划 Agent**：只能从已检索候选中组合指定数量的菜和汤，并调用确定性校验。
4. **食材库存 Agent（按需）**：有现有食材、图片识别或预算时，核对候选覆盖和采购缺口。
5. **烹饪调度 Agent（按需）**：只有明确限时/并行诉求，或多菜与有限设备约束组合出现时才加入，基于食谱步骤证据安排顺序。

研究与饮食分析可以并行；库存分析在检索候选后运行，菜单规划等待证据汇合，排期在菜单确定后运行。设备确认、版本校验和 Mock
执行属于确定性系统节点，不是自由 Agent。这样拆分的依据是数据权限、并行性
和失败边界，而不是角色数量。

## 一条主链

```text
Web/IM 自然语言或 Web 图片
  -> TurnApplicationService（规范化请求、加载短期任务状态）
  -> 普通聊天 / 厨房技巧 / 搜索继续走共享 Runtime
  -> 图片：视觉识别 -> 用户确认或补充食材 -> 混合 RAG
  -> 菜单路由在同一 Turn 内完成检索；复杂任务在 Recipe handler 前交接 Graph
  -> SearchRequest 转换为 Brief：ComplexityProfile 从菜单/饮食/库存/调度四维决定是否进图和动态组队
  -> LangGraph 复用本轮真实候选：research || dietary -> [inventory] -> menu -> [scheduler] -> validate -> review -> final_validate
  -> interrupt 等待用户确认
  -> 可选“第二道换掉”：创建子 Run，只改目标槽位，再校验
  -> 确认最新版本 -> Mock 执行 -> Trace/事件账本展示
```

图片识别结果不是事实终点。Web 先返回结构化的
`image_ingredients_confirmation`，把食材名称保存为待办；用户回复“再加豆腐”或
“就这些”后才发起检索，避免视觉误识别直接污染推荐。

## 状态与身份关系

| 概念 | 主键/关系 | 作用 |
| --- | --- | --- |
| Web 会话 | `web:<32位随机值>` | 隔离聊天历史、候选、图片待办和任务状态 |
| Graph Run | 独立 `run_id`，同时保存 `thread_id` | 表示一次可追踪的配餐执行 |
| 局部修订 Run | 新 `run_id` + `parent_run_id` | 保留原方案证据，只替换指定槽位 |
| Graph checkpoint | `thread_id:workflow:run_id:v<version>` | 隔离并恢复具体版本的人机确认状态 |
| Mock execution | `run_id:v<version>` | 幂等键；重复确认不会重复制造副作用 |

用户只感知 `thread_id`。系统通过它找到该会话最新的 Graph Run；`run_id`、版本号
和 checkpoint 是内部执行身份。确认时必须同时匹配 Run 状态和版本，旧页面上的确认
不能授权新方案。

`runs` 是任务状态的权威来源，`chat_links` 只保存每个会话的“当前 Run”指针，二者在
同一个 SQLite 事务中创建和绑定。同一会话已有 `running` 或
`awaiting_confirmation` Run 时，重复请求会复用它，不会并发创建两个可确认方案。
Run 还会持久化完整 `checkpoint_thread_id`，因此服务恢复时不再依赖临时字符串拼接。

## P1 恢复与可观测性

- 服务启动会把遗留的 `running` 标成 `interrupted`，避免把已不存在的后台协程展示
  成仍在运行；用户可以显式从同一 checkpoint 重试。
- 重试不创建新 Run、不增加版本，也不会绕过人工确认；它只增加 `attempt` 并记录
  `recovery_start` 事件。若进程在首个 checkpoint 写入前退出，则从持久化 Brief
  安全重建初始输入；Mock 执行仍使用原幂等键。修改方案才增加版本并使用新的 checkpoint。
- `/api/runs/{run_id}` 返回 `workflow_identity` 和基于真实事件聚合的 `metrics`：模型
  调用、工具调用、完成节点、Token、实际运行耗时、错误和尝试次数；另保留包含人工
  等待时间的生命周期墙钟耗时供排障使用。
- 这些本地指标用于演示和故障定位，不等于生产级 OpenTelemetry、跨进程任务队列或
  SLO 监控。生产化仍需独立 Worker、租约/心跳、集中式事件库和告警。

## 统一接管与排期证据边界

复杂菜单不再先生成一份普通菜单回复、再启动另一条 Graph。`workflow_handoff`
位于统一 Turn 编排器的 Recipe handler 前：它复用本轮 Router 的 `SearchRequest` 和
真实 RAG 候选，只让一个 handler 成为本轮响应所有者。Graph 的 Research Agent
仍通过受限工具读取这些候选并复核 ID，但不做第二次初始检索。

公开仓库无法分发生产食谱步骤时，`DEMO_VIRTUAL_SCHEDULE=true` 会给最终菜单注入
明确标记为 `demo_template` 的虚拟准备/烹饪步骤。页面显示“演示模板”和“演示排期”，
它们只用于说明调度算法，不作为真实食谱事实、时限承诺或设备指令。只有
`detail_basis=source` 的来源步骤才允许显示“可核验时间”。最终确定性校验会汇总
菜单、饮食、库存、排期和审阅结果；阻断硬冲突，将虚拟步骤、价格和份量缺口保留为
人工确认项。

## RAG 后端

- `hybrid_rag`：复用主工程检索服务，执行 dense + BM25、RRF 融合和 rerank。
  普通聊天与动态选择的 Agent 共享这个后端，检索失败时不会生成虚构候选。
- `public_rehearsal`：公开仓库的可运行规则演练库，用于没有授权数据的克隆环境。
  它会在页面中明确标注，不作为混合 RAG 成果证据。
- `DEMO_RECIPE_BACKEND=auto`：检测已配置的 Milvus 地址；存在则优先真实混合 RAG，
  否则使用公开演练库。面试前建议显式设为 `hybrid`，缺库时直接失败，避免误演示。

## 面试演示脚本

演示前先运行离线契约自检。它只使用明确标记的虚拟食谱，不访问模型、Milvus 或设备：

```bash
uv run python scripts/verify_interview_demo.py
```

1. 先问一句厨房技巧，说明普通问题不必进入复杂 Graph。
2. 上传一张冰箱食材照片，展示识别清单，回复“再加豆腐”，然后回复“就这些”。
3. 直接发送“今晚四个人吃，用鸡肉、西兰花和鸡蛋规划三菜一汤，不要花生；只有一个灶台和空气炸锅，希望 45 分钟内同时做好，采购预算 50 元”。
4. 展示四维复杂度、动态选择的五个角色、research 与 dietary 并行、共享状态、库存覆盖、
   烹饪排期、菜单校验和 RAG 后端标识。
5. 在确认前说“第二道换清淡一点”，说明只重跑必要分支并生成新的 Run。
6. 回复“确认”，展示最新版本校验、Mock 执行结果和 Trace。

这条演示能回答四个核心问题：为什么使用多 Agent、RAG 证据从哪里来、会话与运行
如何隔离、以及有副作用动作为什么不能直接交给模型。

## 当前公开边界

- Web 对话状态使用进程内存；Graph Run、事件、checkpoint 和 Mock 执行账本使用本地
  SQLite。服务重启后应新建匿名 Web 会话。
- 视觉与 live Agent 依赖本地 DashScope 配置和网络，自动化测试使用替身验证协议，
  不把替身结果描述为真实模型效果。
- 真实食谱数据、生产 Milvus、用户账号和硬件厂商适配器不随公开仓库分发。
- 公开数据缺少步骤时展示的是明确标记的虚拟排期，不代表生产食谱做法或真实设备能力。
