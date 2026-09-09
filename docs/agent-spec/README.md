# CookClaw Agent Reply & Search Specification

> 审计基线：2026-07-17；架构事实已于 2026-08-05 按单一编排根同步
> 范围：Agent、Prompt、Intent、Router、Search/RAG、Device、Memory、Image、Multi-channel、Error Handling、Reply Style
> 当前性质：AS-IS 章节随代码更新；TO-BE 和 TODO 不代表已经通过外部验收。

## 如何使用这套文档

本文档同时保留两种视角：

- **AS-IS（当前事实）**：以当前仓库代码实际执行路径为准。注释、README 或旧 Prompt 与代码冲突时，代码优先。
- **TO-BE（目标规范）**：作为后续实现、测试和评审的产品规则，不代表当前已经落地。

规范词含义：

- **MUST**：违反即产品错误、安全事故或数据可信度问题。
- **SHOULD**：通常必须遵守；只有明确、可解释的例外才能偏离。
- **MAY**：可选优化，不影响基础正确性。

## 文档目录

| 文档 | 内容 |
|---|---|
| [00-Agent-Overall-Architecture-Map.md](00-Agent-Overall-Architecture-Map.md) | Agent 整体架构图、调用链速查、开关与红线 |
| [01-Agent-Architecture.md](01-Agent-Architecture.md) | 真实架构、主链、状态机、文件定位 |
| [02-Reply-Rules.md](02-Reply-Rules.md) | 当前回复规则与目标 Reply Rule Specification |
| [03-Intent-Spec.md](03-Intent-Spec.md) | 意图分类、隐式子意图、目标槽位协议 |
| [04-Search-Rules.md](04-Search-Rules.md) | 搜索决策树、上下文追问、Query Rewrite 规则 |
| [05-RAG-Spec.md](05-RAG-Spec.md) | Query → Hybrid Recall → Rerank → Selection 全流程 |
| [06-Tool-Spec.md](06-Tool-Spec.md) | 搜索、视觉、设备、记忆、通道工具契约 |
| [07-Memory-Spec.md](07-Memory-Spec.md) | 进程态、会话记忆、账号画像与清除规则 |
| [08-Error-Handling.md](08-Error-Handling.md) | 异常分层、错误码、用户回复与降级策略 |
| [09-MultiChannel-Spec.md](09-MultiChannel-Spec.md) | Web、QQ、WhatsApp、微信的一致性与差异 |
| [10-Reply-Templates.md](10-Reply-Templates.md) | 可复用但不机械的中英文回复模板 |
| [11-Agent-Improvement-Report.md](11-Agent-Improvement-Report.md) | 成熟度评分、风险、冲突与 Top 20 改进项 |
| [TODO.md](TODO.md) | P0/P1/P2 落地清单与验收条件 |

## 事实源优先级

出现冲突时按以下顺序判断当前行为：

1. `app/main.py`、`app/api/routes/chat.py` 的真实入口与通道处理；
2. `app/orchestrator/turn/` 的唯一组合根、Turn 协议、Handler adapter 和 Renderer；
3. `app/orchestrator/planning/` 与 `app/orchestrator/` 顶层领域路由；
4. `app/agent/skills/recipe-search/`、`app/agent/skills/recipe-operation/` 的工具实现；
5. `app/conversation/` 的记忆/任务态实现与 `app/ports/` 的抽象端口；
6. `app/core/*.md` 的当前 Prompt；
7. README、`docs/` 旧文档和代码注释。

旧文档与 Prompt 仍有审计价值，但不能单独证明某条规则正在生产主链执行。

## 不可突破的红线

1. 菜谱名称、ID、图片、食材、标签等菜谱事实只能来自真实检索结果或已保存的真实检索快照。
2. 视觉模型只生成识别线索和检索词，不能直接生成可执行菜谱或触发设备。
3. 设备操作必须绑定到确定的用户、设备和菜谱，并在真实开火前取得明确确认。
4. 工具失败、状态未知或数据缺失时必须如实说明，不能用语言流畅度掩盖事实缺口。
5. 历史偏好只能补充本轮需求，不能覆盖用户当前明确条件、忌口或纠正。
