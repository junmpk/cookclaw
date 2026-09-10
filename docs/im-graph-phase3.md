# IM 通道 Graph bridge：第三阶段

## 已实现范围

QQ、微信和 WhatsApp 的文本消息本来就共同进入 `qqbot_chat()` 与统一 Turn
Orchestrator。本阶段在这个共享入口增加可选的 `IMGraphBridge`：

1. 普通闲聊、厨房技巧、单菜搜索继续走原 Turn Runtime；
2. 只有明确包含“三位专家协作”“协作配餐”等触发词的菜单请求才进入 Graph；
3. 原 Runtime 先解析并保存人数、菜数、汤数和排除项，Graph 不从自由文本重新猜约束；
4. Graph 完成后返回统一 `menu_plan`，三个通道沿用现有菜谱卡片 Renderer；
5. 同一通道 thread 可以回复“确认”“取消”或局部调整菜单。

## 局部调整

以下表达会定位到具体菜单槽位：

- `第二道换掉`
- `把第二道换成西兰花烤土豆`
- `第一道清淡一点`
- `dish 2 replace`

系统重新检索时会带上调整原话，并在真实候选中优先选择与该要求更相关的同类型
食谱。只有目标槽位允许变化，其余菜位和汤保持不变；新方案必须重新经过数量、
重复项、排除食材和来源校验。如果用户只说“这道菜换掉”而没有序号，系统会要求
补充第几道，不猜测目标。

“清淡”等主观要求只能用于检索与候选排序；当源食谱缺少油盐用量时，系统不能
声称已经满足精确营养目标。

## 本地启用

先准备公开演示食谱库：

```bash
uv run python -m app.demo.seed
```

然后在本地 `.env` 增加：

```dotenv
MULTI_AGENT_BRIDGE_ENABLED=true
MULTI_AGENT_BRIDGE_MODE=live
# 可选；默认使用仓库根目录的 .demo
# MULTI_AGENT_DATA_DIR=/absolute/path/to/cookclaw/.demo
```

启动主应用后，已启用的 QQ、微信和 WhatsApp 通道会共享此 bridge：

```bash
uv run python -m app.main
```

测试时可把 `MULTI_AGENT_BRIDGE_MODE` 改成 `rehearsal`。它沿同一张 LangGraph
运行，但不调用模型，且必须明确标注为规则演练。

## 安全与工程边界

- 开关默认关闭，未启用时不会改变任何通道行为；
- Graph 只产生菜单方案，执行节点仍然只有 Mock provider；
- 确认前后都会校验版本和菜单约束，Agent 没有设备授权；
- Graph checkpoint、thread 与 run 的绑定保存在本地 SQLite；
- 当前只完成本地自动化验证，没有连接真实 QQ、微信或 WhatsApp 账号验收；
- 当前同步等待 Graph 结果，适合面试演示；生产环境应改为持久任务队列和主动回推。

## 面试讲解重点

这一阶段体现的不是“再加三个通道 Agent”，而是把通道协议与业务编排解耦：

`QQ / 微信 / WhatsApp → qqbot_chat → Turn Runtime → 条件 Graph bridge → LangGraph`

通道只负责身份、媒体和消息投递；约束解析、Graph 状态、局部修订与确认语义均在
共享业务层实现，因此同一条测试可以覆盖多个通道。
