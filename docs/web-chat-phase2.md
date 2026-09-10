# Web 共享对话：第二阶段

## 目标

第二阶段把三 Agent LangGraph 作为共享 Web 对话中的一个受限配餐子图：

- 普通闲聊、厨房技巧、菜谱搜索和详情继续走共享 Turn Runtime；
- 用户明确说“请让三位专家协作配餐”等触发语句后，先由共享核心解析并保存
  SearchRequest，再把已确认的菜/汤数量、人数和排除项转换成 Graph Brief；
- Graph 负责食谱研究、饮食分析、菜单规划、校验/修订和人工确认；
- 用户仍在同一个 thread_id 中回复“确认”或“取消”，不需要复制内部 run_id。

## 阅读顺序

1. app/demo/web/app.js：聊天发送后轮询 Graph 任务，并复用原有 Trace/确认视图。
2. app/demo/chat_intent.py：Graph 触发、确认和局部修订的确定性解析。
3. app/demo/server.py：Web `/api/chat` 的 bridge 分支；Graph 任务通过
   `DemoStorage.chat_links` 绑定聊天线程。
4. app/agent/participle_agent.py：共享核心解析自然语言并保存任务状态。
5. app/demo/graph.py：LangGraph 的并行节点、条件修订、interrupt 和执行前校验。
6. app/demo/agents.py：三个 Agent 的最小工具权限与 Pydantic 结构化输出。

## 演示方式

先在左侧聊天发送：

> 请让三位专家协作规划：今晚两个人吃，有鸡肉和西兰花，两菜一汤，不要花生。

页面会先显示共享核心解析结果，然后展示 LangGraph 的研究、饮食分析、菜单规划
和校验事件。图运行进入人工确认后，在同一个聊天框发送“确认”或“取消”。

在确认前也可以发送“第二道换掉”“把第一道菜换一下”“第一道清淡一点”等
局部修订指令。
系统会保留未修改的菜位和汤，创建新的 Graph 版本并重新校验。普通消息不会触发
Graph。也可以使用左侧快捷按钮“三 Agent 配餐”。

## 边界与取舍

- bridge 只接受共享状态中已经形成的显式菜单请求；缺少菜/汤数量时继续走普通聊天，
  不让 Graph 自行猜测数量。
- 局部修订只按槽位替换同类型候选，不接受模型或用户提供的候选外 recipe ID；
  没有可核验替代项时，Graph 会进入 blocked，不拿重复菜凑数。
- Graph 的执行仍是 Mock；设备动作不属于 Agent 工具权限。
- chat_links 与 Graph checkpoint 保存在本地 Demo SQLite，公开入口的对话状态仍使用
  InMemoryConversationStore，重启后建议新建聊天线程。
- Graph 的 live 模式会调用 DEMO_MODEL；测试和流程演示可以显式传 mode=rehearsal，
  沿同一张图运行但不调用模型。
- 第三阶段已经提供可选的 QQ/微信/WhatsApp Graph bridge，默认关闭；启用方法和
  本地验证边界见 `docs/im-graph-phase3.md`。

## 学习重点

- 用一个稳定的 SearchRequest 在“自然语言理解”和“工作流编排”之间传递边界；
- 用 StateGraph 表达并行、条件边和人工确认，而不是把所有消息都交给 Agent；
- 用 SQLite link、checkpoint 和版本号实现“同一聊天线程继续确认”；
- 对 Graph 的输入做 Pydantic 校验，对执行前状态再次做确定性校验。
