# 11. CookClaw Agent 改进报告

> 当前快照：2026-08-05
> 适用分支：`refactor/agent-planner-orchestrator`
> 说明：原 2026-07-17 审计中的数字评分缺少同环境线上样本，已停止沿用；成熟度以可验证退出条件判断。

## 1. 当前结论

CookClaw 已从“多个入口分支 + 旧完整 Agent + 局部规则”演进为：

> 统一 Turn facade + Bounded Planner + 显式领域 Handler + 确定性设备状态机

本地内部改造已收敛为单一 Turn 主链，Planner 默认 `TURN_PLANNER_MODE=off`。没有测试服务器、真实通道、
多 worker 和真实设备证据前，不能把“本地实现完成”写成“生产验收完成”。

## 2. 已确认的当前能力

| 能力 | 当前实现 | 代码事实源 |
|---|---|---|
| 单轮编排所有者 | 固定 Handler 顺序，首个命中结束本轮 | `app/orchestrator/turn/turn_orchestrator.py` |
| 低风险规划 | 结构化 `TurnPlan`、Validator、单步 Executor、可选评估门禁 | `app/orchestrator/planning/` |
| 菜谱真实性 | Milvus 混合检索和确定性 grounding；失败不由模型补菜谱 | `app/agent/recipe_search_service.py`、`app/agent/fast_path.py` |
| 设备安全 | pending、明确确认、原子 action 领取、durable 未决执行态、禁止自动重发 | `app/orchestrator/cook.py`、`app/conversation/task_state_store.py` |
| 会话与画像 | Redis 活跃会话/任务态，PostgreSQL 长期画像 | `app/conversation/` |
| 状态边界 | 主编排通过纯 `DialogueStatePort`，具体工作区归 conversation | `app/ports/dialogue_state.py`、`app/conversation/dialogue_state_workspace.py` |
| 多通道核心 | Web 与三个 IM 固定共用唯一 Turn 组合根 | `app/orchestrator/turn/facade.py`、`response_renderer.py` |
| 图片链 | 视觉结果进入 `DerivedTurnContext`，只允许 grounded 菜谱动作 | `app/orchestrator/turn/image_handler.py` |
| 可观测性 | 脱敏 Trace、Planner/Handler/Tool/安全汇总、显式阈值告警 | `app/observability/`、`scripts/summarize_conversation_traces.py` |
| 旧攻击面清理 | 旧完整 Agent、MemorySaver、shell/skills backend 和旧总 Prompt 已删除 | `app/agent/participle_agent.py`、阶段 5.1 回归 |

## 3. 相比原审计已解决的问题

### 3.1 架构和流程

- 新增统一 `TurnRequest / TurnContext / TurnPlan / ResponseEnvelope`；
- Memory、Recipe、Device 形成显式 Handler 和领域结果；
- Web、QQ、微信、WhatsApp 固定接入同一 facade；
- 图片不再伪造成用户文本，而是带来源派生上下文；
- active Planner 只执行白名单中的单个低/中风险步骤；
- 已发生工具调用或状态变化后的执行失败会 fail closed，不重新跑旧链。

### 3.2 安全和真实性

- Planner action 集合不含启动、停止、确认和记忆写入；
- Planner 参数不能覆盖用户原话中的搜索条件或设备语义证据；
- 设备启动仍由确定性 pending/确认链拥有；
- 搜索失败不能进入自由模型生成菜谱；
- 存储结果通过结构化状态表达，失败不能伪装已保存；
- Trace 不记录原始用户消息、完整 thread、设备 ID、菜谱正文或工具 payload。

### 3.3 维护和回溯

- 每个迁移阶段均有独立 Git 提交和阶段记录；
- Orchestrator 只保留统一主链，代码版本是唯一回滚边界；
- Planner 支持 `off/shadow/active`，active 再按账号和稳定百分比分桶；
- 健康接口显示当前非敏感运行模式；
- Trace 汇总可对 Planner fallback、执行失败、工具失败和设备红线做机器判断。

## 4. 尚未解决或尚未验收

| 级别 | 项目 | 当前事实 | 完成条件 |
|---|---|---|---|
| P0 | 账号—设备 ACL | 当前设备映射仍是 Demo 边界 | 明确租户、账号、设备、角色、授权和撤权事实源并全链校验 |
| P0 | 真实设备验收 | 本地测试使用 mock | 验证 action ID、重复确认、启动/停止、超时和 unknown 状态 |
| P0 | 多 worker 一致性 | Port/Redis 接缝已存在，未做故障注入 | CAS、TTL、重启、网络分区和恢复测试通过 |
| P0 | 通道管理面安全 | 各通道接入机制不同，本文未完成真实公网安全审计 | webhook 来源验证、管理接口鉴权、防重放和网络边界通过 |
| P1 | Planner 产品收益 | 本地只证明协议与安全边界 | 同模型、同库、同设备盲测优于 legacy，硬指标不下降 |
| P1 | Planner 产品策略 | 默认 off，仍保留独立 shadow/active 评估能力 | 用真实样本决定固定关闭或启用，不影响统一 Turn 主链 |
| P1 | 大文件继续拆分 | `participle_agent.py` 仍承载较多领域 handler | 按领域迁出剩余 handler，保持公开协议不变 |
| P1 | 通道展示一致性 | 核心 Envelope 已统一，真实客户端未全量回放 | 四通道相同案例结论一致，媒体部分失败有结构化结果 |
| P2 | 受控 Deep Agent 成本 | 只读边界明确，真实收益/成本尚未冻结 | 用 A/B 决定保留、缩小或替换 |

## 5. 当前体验问题如何判断

不能再只用“感觉更自然”评价架构。每个测试轮至少关联：

- 用户目标和上下文是否理解正确；
- Planner action、phase 和 fallback 是否合理；
- 工具是否该调、是否只调一次、参数是否有证据；
- 菜谱事实是否来自真实候选；
- 设备是否停留在正确的 pending/执行状态；
- 回复是否承接上文、只问一个必要问题、给出自然下一步；
- 延迟、token、超时和失败类型。

测试人员负责盲测和主观评分；测试服务器必须负责保存脱敏 Trace。只看最终回复无法判断
OpenClaw/CookClaw 的流畅感来自正确规划，还是来自更宽松的自由编造。

## 6. 放量硬门槛

以下任一出现，版本不得继续放量：

1. 模型生成不存在的菜谱事实；
2. 同一轮重复发送同类设备命令；
3. Planner 执行启动、停止、确认或记忆写入；
4. 无有效 pending/确认仍下发设备动作；
5. 用户或通道间状态串线；
6. 工具结果未知却回复已成功。

成功率、P95、超时率、工具失败率和 Planner fallback 不能在代码里拍脑袋固定。应先采集
同模型、同向量库、同设备环境的基线，再由产品/架构责任人冻结阈值。

## 7. 推荐验收顺序

1. 部署 `all + shadow`，确认四通道不改变原工具次数和回复；
2. 测试人员执行统一场景，服务端拉取 Trace 并补人工标签；
3. 只对白名单账号开启 active，百分比保持 0；
4. 先验证对话、搜索、推荐和详情，再验证设备预检；
5. 设备启动/停止使用专用测试设备，核对命令回执与真实状态；
6. 冻结阈值并运行 `--fail-on-alert`；
7. 完成回滚演练后才逐步提高稳定分桶；
8. 观察期结束后再决定删除 legacy facade。

## 8. 结论

这次重构已经把“聪明感”放到受约束规划和自然表达，把菜谱事实、状态和设备副作用留在
确定性内核。内部开发阶段的下一步不再是继续增加路由分支或复制 OpenClaw 的自由循环，
而是用真实 Trace 和盲测证明新链在自然度、正确率、成本和安全上确实优于基线。
