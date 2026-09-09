# 06. Tool Specification

## 1. 工具边界

CookClaw 当前存在三类“工具”，审计时必须区分：

1. **在线运行时工具**：主链实际调用的搜索、视觉、设备、记忆和通道能力；
2. **受控 Agent 只读工具**：只读取当前裁剪状态和真实候选，不执行搜索或设备动作；
3. **离线工程工具**：灌库、标签补全、翻译、评测脚本，不应被终端用户直接触发。

把“仓库里存在”误写成“线上会调用”，会导致行为规范失真。

## 2. 在线工具清单（AS-IS）

| 工具 | 用途 | 核心输入 | 核心输出 | 主要失败 |
|---|---|---|---|---|
| Recipe Search | 从 Milvus 检索真实菜谱 | query、lang、limit、filters、exclude | recipe id/name/image/tags/ingredients 等 | Milvus、Embedding、Rerank、子进程失败 |
| Vision Analyze | 从图片识别菜名/食材线索 | 本地图片、用户文字、语言 | 识别文本、检索关键词 | 图片下载、格式、VL 服务、低置信度 |
| Candidate Resolver | 从最近候选中比较/选择/解释 | candidate snapshot、用户条件 | 限定候选内的选择和事实解释 | 快照过期、指代不清 |
| Controlled Conversation Agent | 低风险推荐表达与候选内选择 | 裁剪任务状态、真实结果、近期用户原话、当前问题 | 经外层事实校验的推荐草稿或 recipe id | 模型超时、越界 ID、未调用只读工具 |
| Device Check | 查询已配置设备及状态 | 设备配置/设备 ID | 在线、忙闲、运行状态 | 配置缺失、网络、鉴权、超时 |
| Device Start | 启动真实菜谱 | device id、真实 cookId、`start` | API code、启动结果 | 离线、忙、拒绝、状态未进入运行 |
| Device Stop | 停止设备 | device id、`stop` | API code、停止结果 | 无活动任务、设备错配、状态未验证 |
| Conversation Memory | 读取/写入近期上下文、偏好、搜索历史 | thread/profile key、turn/event | recent turns、summary、preferences、snapshots | Redis/PG 超时、版本冲突、过期 |
| Channel Sender | 将统一回复渲染并发送到各 IM | text/cards/media、recipient | message id/send status | 令牌、限流、媒体上传、网络失败 |

## 3. Recipe Search 契约

- 输入 MUST 是标准化 SearchRequest 派生的查询，保留原始用户文本用于审计。
- 输出中的菜谱事实 MUST 来自 Milvus 文档。
- 失败时先走进程内 → 子进程降级；两者都失败则返回明确错误。
- 不得把空结果替换为模型生成菜谱。
- 搜索成功后 MUST 保存候选快照，支持“这三道”“换一批”“上一批”。

## 4. Vision 契约

- 视觉结果只是线索，不是菜谱事实。
- 低置信度时 SHOULD 说明看不清，并只追问一个关键问题。
- 视觉识别后若要推荐菜谱，MUST 再调用真实 Recipe Search。
- 图片不得直接触发设备启动。
- 下载、上传和临时文件处理需要大小、类型、域名和超时限制。

## 5. Device 工具契约

### 5.1 启动前

MUST 同时满足：

1. 用户身份已绑定并授权目标设备；
2. `cookId` 来自当前或可追溯的真实候选；
3. 设备在线且不处于冲突状态；
4. 用户明确确认“设备 + 菜谱 + 开始”；
5. 命令具备幂等键，避免重试导致重复启动。

### 5.2 启动后

- API 返回 200 只代表请求被接受，不等于设备已经运行。
- MUST 轮询真实状态确认进入运行；确认失败时回复“状态未确认”，不能说“已经开始”。
- 保存活动任务时，状态有效期应覆盖整个烹饪周期，不应固定 10 分钟后丢失。

### 5.3 停止

- MUST 只停止该账号已授权且与当前活动任务绑定的设备。
- 没有活动任务时不得默认向某台设备发停止命令。
- 停止后 SHOULD 验证设备进入空闲/停止状态。

## 6. Memory 工具契约

- 读取时遵循“本轮明确要求 > 安全偏好 > 长期偏好 > 最近行为弱线索”。
- 写入时记录来源、时间、通道和置信度。
- “我没说过”“忘掉这个”必须撤销对应事实，不只写一条相反记忆。
- 清除后不得在同一回复里继续引用已清除内容。
- 记忆不可作为设备授权依据。

## 7. 通道工具契约

- 业务层输出结构化内容，通道层只负责能力适配。
- 文本发送成功、图片发送失败时，必须保留菜名和关键信息，不能整条丢失。
- 每个通道都应返回可追踪的发送状态；失败时记录 trace id，但用户消息中不暴露内部堆栈。
- 机器人 webhook、二维码、登录、启动等管理接口必须认证。

## 8. 受控 Agent 与离线工具

阶段 5.1 已删除旧 Web/IM 完整 Agent、通用 shell/Skill backend、MemorySaver、旧总
Prompt 和无关 self-improving Skill。生产主链遵循：

- 生产主链只开放最小化、显式 schema 的工具；
- 不向面向用户的 Agent 暴露通用 shell 或任意文件系统；
- 未使用的 Agent/Prompt 不留在运行时；
- 灌库、迁移、标签补全、翻译和评测脚本只允许运维/离线角色执行。

新增的 `controlled_deep_agent.py` 不复用旧 Skill，只注册以下只读业务事实：

- `get_current_conversation_state`：读取裁剪后的当前任务、`SearchRequest`、选择、排除和待确认阶段；
- `get_current_recipe_search_results` / `get_current_recipe_candidates`：读取 Router 已完成检索和硬过滤后的真实结果，不重复调用 Milvus；
- `get_verified_recipe_detail`：只允许按本轮结果 ID 读取已有结构化详情，候选外 ID 明确失败。

内建文件、execute、todo 和 subagent 调用由白名单中间件阻断，文件读写另有
deny-all 权限。推荐草稿继续经过 `recommendation_response.py` 的逐句事实校验；
候选 ID 必须再次验证属于当前结果。任何异常、越界输出或超时都回退事实安全的
确定性链路，不再追加第二次候选选择模型调用。设备查询、启动、停止和确认工具均
未注册给受控 Agent。

## 9. 统一工具返回协议（TO-BE）

~~~json
{
  "ok": false,
  "tool": "device_start",
  "trace_id": "...",
  "data": null,
  "error": {
    "code": "DEVICE_STATE_UNVERIFIED",
    "retryable": true,
    "user_message_key": "device.start.unverified",
    "detail": "仅写入受控日志"
  }
}
~~~

统一协议 SHOULD 包含：超时、重试次数、依赖耗时、降级路径、结果来源和安全审计字段。业务层根据错误码选回复，不能解析不同脚本的自然语言输出。

## 10. Tool 调用原则

- 能用确定性工具完成的任务，不交给模型猜。
- 写操作比读操作需要更严格的身份、确认、幂等与结果验证。
- 多 Tool 流程必须有中间状态；任一步失败都不能假装完整成功。
- Planner MAY 用于复杂菜单/多设备编排，但不能绕过搜索真实性和设备确认红线。
