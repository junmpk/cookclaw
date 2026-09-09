# 03. Intent Specification

> 当前快照：2026-08-01

## 1. Intent 不是唯一决策中心

CookClaw 当前有三层不同职责：

1. **确定性 exact/state rule**：处理 pending 确认、取消、停止、进度、序号等上下文动作；
2. **轻量 Intent classifier**：规则未命中时输出粗业务分类、动作和搜索槽位；
3. **Bounded Turn Planner**：命中 active 灰度时选择低风险单步和回复动作。

设备副作用最终仍由确定性 Handler 授权。Intent 或 Planner 的模型输出都不能直接启动、
停止、确认设备或写长期记忆。

## 2. IntentCategory（AS-IS）

定义：`app/orchestrator/intent.py`

| Category | 含义 | 默认业务去向 |
|---|---|---|
| `recipe_search` | 按明确菜名/条件找菜谱 | 真实 RAG |
| `recipe_recommend` | 让助手按场景做选择 | 真实 RAG + grounded 推荐 |
| `recipe_execute` | 设备预检、确认、停止或进度语义 | Device Handler/状态机 |
| `device_manage` | 用户设备实例只读查询 | Device status |
| `cooking_qa` | 技巧、替换、调味、保存等 | 单次无工具 QA |
| `greeting` | 问候或能力介绍 | Identity/Greeting Handler |
| `off_topic` | 非烹饪普通话题 | 受控小聊或边界回复 |
| `unknown` | 解析失败或未支持类别 | 保守 QA/smalltalk/澄清 |

未知分类不会伪造置信度；`IntentResult.confidence` 保持 `None`。即使未来增加置信度，也
不能用于设备授权。

## 3. Classifier 输出（AS-IS）

Prompt：`app/core/intent_check_prompt.md`

```json
{
  "r": true,
  "c": "recipe_recommend",
  "a": "new_search",
  "k": ["鸡肉", "清淡"],
  "slots": {},
  "s": {
    "q": "清淡 鸡肉",
    "dish": [],
    "ingredient": ["鸡肉"],
    "cuisine": [],
    "flavor": ["清淡"],
    "method": [],
    "scene": [],
    "meal": [],
    "diet": [],
    "exclude": [],
    "avoid": []
  }
}
```

`s` 只用于搜索/推荐。解析后生成强类型 `IntentResult` 和 `SearchRequest`；未知 action 会
被 `infer_classifier_action()` 归一化，不会动态映射成任意函数。

## 4. RouteDecision（AS-IS）

规则和分类器统一输出：

```text
category
action
source = exact_rule | state_rule | classifier | parse_error | ambiguous
reason_code
risk = low | medium | high
slots
context_ref
needs_clarification
classifier_called
trace_id
```

日志用 `context_ref`，只包含 pending 是否存在、候选数量、约束数量和消息类型；不记录
完整用户原话、菜谱、设备或状态 payload。

## 5. 确定性动作优先级

`app/orchestrator/routing_policy.py:decide_exact_route()` 在调用分类器前处理：

1. pending device 下的取消、确认和设备选择；
2. active cooking 下的停止和进度；
3. 无 active task 的停止/进度歧义；
4. 有候选时的序号选择；
5. 有候选时的换批；
6. 一般 pending 的取消；
7. 无 pending 的裸确认；
8. 无任何上下文的“继续”。

这些动作需要真实状态才能解释，不能交给模型凭字面猜测。

## 6. Classifier action 集合

当前允许的粗动作：

```text
new_search
refine_search
compare_candidates
choose_candidate
device_status
select_device
confirm_start
prepare_device
progress
stop
memory_correct
cooking_qa
casual
unknown
```

风险映射：

- `confirm_start / stop`：high；
- `choose_candidate / select_device / prepare_device`：medium；
- 其余：low。

风险标签用于路由和审计，不等于授权。`confirm_start` 和 `stop` 仍必须命中真实 pending/
active 状态与确定性设备流程。

## 7. Planner 与 Intent 的关系

Planner 不替换安全规则，也不直接采用分类器的所有字段：

- exact/device pending 在 Planner 之前执行；
- active Planner action 还需 Validator、白名单和原话语义兼容检查；
- Planner 的 recipe query 不能覆盖原始 `SearchRequest`；
- Planner 执行失败且无影响时，只回退同一个确定性链一次；
- 已有工具调用或状态变化时 fail closed；
- Shadow 只比较，不改变 Intent/RouteDecision 或用户回复。

## 8. 典型判定

| 输入与状态 | 当前动作 | 是否调用模型决定副作用 |
|---|---|---:|
| “家里有鸡腿，找不辣的菜” | new_search | 否，模型可结构化条件，RAG 执行 |
| “朋友来吃饭，推荐四道” | recipe_recommend/menu plan | 否，候选来自 RAG |
| “这三道哪道省事” + 有候选 | compare_candidates | 否，先用候选状态 |
| “第二个” + 有候选 | choose_candidate | 否，状态规则定位 |
| “第二个” + 无候选 | ambiguous | 否，只追问 |
| “确认” + 有 pending start | confirm_start | 否，确定性确认链 |
| “确认” + 无 pending | ambiguous | 否，不猜确认对象 |
| “停止” + 有 active task | stop | 否，确定性 stop |
| “停止” + 无 active task | ambiguous/no-op | 否，不发送命令 |
| “设备在线吗” | device_status | 否，调用只读工具 |
| “谢谢” | casual | 可用单次无工具表达 |

## 9. 当前缺口

- 规则、classifier action、Planner action 是兼容迁移中的三个协议，仍需长期收敛命名；
- sensitive/privacy 尚未形成完整业务 action；
- locale 主要保证中英文，更多语言缺少等价评测；
- 真实多轮误路由率和 Planner action 正确率尚无测试服务器基线；
- `router.py` 仍较大，分类、搜索计划和兼容结果形状可继续拆分。
