# CookClaw P0 用户与记忆存储方案

> 状态：已实施；数据库、应用接入、有效数据迁移与存储切换均已验证
>
> 制定日期：2026-07-23
>
> 数据库 migration：[001_create_channel_users.sql](../../db/migrations/001_create_channel_users.sql)

## 1. 需求复述与验收

当前阶段不建设统一用户体系，不做 QQ、微信、WhatsApp 的用户合并。每个通道身份独立保存：

- PostgreSQL 保存用户基础信息、长期记忆和聚合画像。
- Redis 保存当前会话的短期记忆、候选、澄清和待确认动作。
- Milvus 继续只负责真实食谱检索，暂不保存用户语义记忆。
- 后续有统一账号体系后，再增加统一用户和通道绑定模型。

本阶段验收标准：

1. 相同 `channel + channel_account_id + channel_user_id` 只能有一条用户记录。
2. QQ、微信、WhatsApp 的相同字符串用户 ID 不会自动合并。
3. 用户画像和长期记忆具有合法 JSON 结构及版本号。
4. “今天少辣”等临时条件只进入 Redis；稳定、明确的长期声明才进入 PostgreSQL。
5. PostgreSQL 表结构可重复执行，不破坏已有记录。

## 2. 实施前基线与当前状态

| 能力 | 实施前 | 2026-07-23 切换后 |
| --- | --- | --- |
| 短期会话 | SQLite `ConversationMemory` | Redis `RedisConversationStore`，24 小时 TTL |
| 长期画像 | SQLite 的 `profile:*` 伪会话 | PostgreSQL `public.channel_users` |
| 搜索候选与最近轮次 | SQLite payload | Redis 会话 JSON |
| 长期偏好、摘要、饮食事件 | SQLite payload | PostgreSQL `profile` + `long_term_memory` |
| 长期事实来源与删除状态 | 未单独保存 | `long_term_memory.facts` |
| 设备 pending/执行状态 | 进程内字典 | Redis 独立 task-state key；单轮仅使用隔离工作区 |
| Milvus | 真实食谱检索 | 保持不变，不保存用户记忆 |

SQLite 文件仍保留，只作为迁移源和紧急回滚路径，不再是当前 `.env` 的默认运行存储。

## 3. 用户与数据流

```text
通道消息
  → channel + channel_account_id + channel_user_id
  → 查询或创建 channel_users
  → 从 Redis 读取当前 thread 的短期记忆
  → 从 PostgreSQL 读取 profile 和 long_term_memory
  → 按优先级合并：
       本轮明确输入
       > Redis 当前会话条件
       > PostgreSQL 长期画像
  → 执行真实食谱检索或其他领域任务
  → 更新 Redis 当前会话
  → 回复完成后提取稳定长期事实
  → 乐观锁更新 PostgreSQL
```

## 4. 复用与新建

### 4.1 复用

- 复用 `ConversationMemory` 中的近期轮次、摘要、偏好、搜索历史和饮食事件语义。
- 复用现有 `ConversationStore` 接口，后续增加 Redis 实现。
- 复用现有 `thread_id` 构造规则，保证 QQ、微信、WhatsApp 会话不串线。
- 复用现有真实 Milvus 食谱检索和设备预检链路。

### 4.2 新建

- PostgreSQL `public.channel_users`。
- `PostgresChannelUserProfileStore` 长期画像 Repository。
- `RedisConversationStore` 短期会话 Store。
- SQLite → PostgreSQL/Redis 幂等迁移、核验和写读探针脚本。
- FastAPI 启动健康检查和关闭连接回收。

## 5. PostgreSQL 技术设计

### 5.1 表定位

表名使用 `channel_users`，而不是提前占用统一 `users`：

```text
channel_users
  = 当前分通道身份
  + 当前有效画像
  + 长期记忆事实
```

以后建立统一用户体系时，可以新增：

```text
users
user_channel_bindings
```

再把现有 `channel_users` 作为通道身份来源迁移，不需要推翻长期记忆内容。

### 5.2 唯一身份

```text
UNIQUE(channel, channel_account_id, channel_user_id)
```

- `channel`：`qq`、`weixin`、`whatsapp`、`web`。
- `channel_account_id`：CookClaw 接入的平台机器人账号。
- `channel_user_id`：用户在该机器人账号下的平台用户标识。

微信多机器人必须传真实 `account_id`；QQ 和 WhatsApp 单账号阶段可以使用 `default`。

### 5.3 `profile`

`profile` 保存当前生效、推荐链路可直接读取的聚合结果：

```json
{
  "likes": ["牛肉", "清淡"],
  "dislikes": ["香菜"],
  "allergens": ["花生"],
  "dietary_constraints": ["少盐"],
  "health_goals": ["控钠"],
  "tags": ["偏清淡", "牛肉偏好"]
}
```

规则：

- `allergens` 和明确忌口作为确定性过滤条件。
- `likes` 和普通标签只能作为软排序信号。
- 本轮明确输入始终高于画像。
- 一次搜索或一次浏览不能自动升级为“喜欢”。

### 5.4 `long_term_memory`

`long_term_memory` 保留事实、来源和状态，用于重新构建画像：

```json
{
  "facts": [
    {
      "id": "mem_54a66a5c3a20c41d57d03d22",
      "type": "preference_dislike",
      "key": "preference_dislike",
      "value": "香菜",
      "source": "user_explicit",
      "source_channel": "qq",
      "source_thread_id": "qq:dm:example",
      "confidence": 1.0,
      "status": "active",
      "created_at": 1784772000,
      "updated_at": 1784772000
    }
  ],
  "summary": "用户偏清淡，不吃香菜。"
}
```

长期记忆写入规则：

| 用户表达 | 存储位置 | 原因 |
| --- | --- | --- |
| 今天少辣 | Redis | 本次会话临时条件 |
| 以后都少辣 | PostgreSQL | 明确稳定偏好 |
| 我花生过敏 | PostgreSQL，并进入硬约束 | 用户明确安全事实 |
| 搜索红烧肉 | 不写长期偏好 | 搜索不等于喜欢 |
| 这次太咸了 | 先记录当前反馈 | 单次反馈不直接升级为长期偏好 |
| 忘掉我不吃香菜 | 将对应 fact 标记为 deleted | 防止继续影响画像 |

删除事实后保留 `status=deleted`，重新聚合 `profile`；P0 不物理删除 fact。

### 5.5 并发控制

本地和测试服务器可能同时访问同一条用户数据，因此写入必须使用 `memory_version`：

```sql
UPDATE channel_users
SET
    profile = :profile,
    long_term_memory = :memory,
    memory_version = memory_version + 1,
    updated_at = NOW()
WHERE id = :id
  AND memory_version = :expected_version;
```

当前实现遇到版本不一致时抛出 `ConversationStoreConflict`，拒绝旧版本覆盖新版本。自动重新读取并合并属于下一阶段增强项。

## 6. Redis 短期记忆设计

P0 先使用一个会话 Key：

```text
cookclaw:demo:conversation:{sha256(thread_id)}
```

建议内容：

```json
{
  "thread_id": "qq:group:example:user",
  "channel": "qq",
  "user_id": "user",
  "recent_turns": [],
  "summary": {},
  "preferences": {},
  "latest_search": {},
  "search_history": [],
  "version": 1,
  "updated_at": 1784772000
}
```

建议 TTL：

- 普通短期会话：24 小时。
- 候选有效期：应用层检查时间戳，默认 10 分钟。
- 设备最终确认：本次尚未迁入 Redis，仍使用现有进程内状态。

Redis 不保存通道密码、Token、长期过敏事实唯一副本或不设 TTL 的完整聊天记录。

## 7. 设备可执行性

用户画像只影响检索过滤和软排序，不能直接修改设备执行参数：

```text
少辣 / 多炖两分钟 / 加辣椒酱
  → 本轮条件、偏好或手动备注
  → 获取真实 recipe_id / cookId
  → 设备详情和兼容性预检
  → 用户最终确认
  → 才允许发送 start
```

没有新的真实 `cookId` 时，不能把参数修改称为可执行定制菜谱。

## 8. 里程碑

### P0-A：数据地基（已完成）

- 创建 `channel_users`。
- 验证唯一身份、JSON 默认值、约束和幂等执行。
- 保留 SQLite 回滚路径。

### P0-B：应用接入（已完成）

- 增加 PostgreSQL 用户 Repository。
- 增加 Redis ConversationStore。
- 实现长期记忆提取、画像聚合和 `memory_version` 乐观锁。
- 增加功能开关，支持 SQLite 回滚。
- 迁移仍有效的 SQLite 数据并逐条验证版本。

### 后续用户体系

- 新增统一 `users` 和通道绑定表。
- 提供绑定、验证、解绑和迁移流程。
- 再决定是否拆分长期记忆、审计和健康信息表。

## 9. 风险与未决

1. 单表 JSONB 适合当前 Demo，但必须限制 facts 数量并定期摘要，避免单行无限增长。
2. 本地和测试服务器共用数据时已经使用乐观锁；发生冲突会拒绝覆盖，但当前尚未自动重试。
3. 同一通道账号不能在本地和服务器同时消费消息，否则可能重复回复；这与存储共享无关。
4. 当前不同通道用户不会共享称呼或偏好，这是本阶段的明确设计结果。
5. 健康信息、完整聊天记录和通道凭证不应为了 Demo 便利全部写入本表。

## 10. 数据库执行记录

### 10.1 执行环境

| 项目 | 结果 |
| --- | --- |
| 执行日期 | 2026-07-23 |
| 数据库 | `cookclaw` |
| 数据库用户 | `cookclaw` |
| PostgreSQL 版本 | 18.3 |
| Schema | `public` |
| Migration | `db/migrations/001_create_channel_users.sql` |
| Verification | `db/verification/001_verify_channel_users.sql` |

### 10.2 实际执行结果

1. 执行 migration 第一次：成功创建 `public.channel_users`、索引和注释。
2. 执行 migration 第二次：成功，确认脚本可重复运行。
3. 最终表结构包含 14 个字段。
4. PostgreSQL 18.3 中登记 20 个约束对象，其中包括主键、唯一约束、7 个业务检查约束和 `NOT NULL` 约束。
5. 最终包含 3 个索引：主键、通道身份唯一索引、活跃用户最近访问索引。
6. 表注释已写入数据库。
7. 建表验证阶段业务数据行数为 0；后续应用切换阶段完成了有效历史数据迁移。

### 10.3 验证结果

验证脚本在事务中完成并最终 `ROLLBACK`：

| 验证项 | 结果 |
| --- | --- |
| 默认 `profile` 为 JSON object | 通过 |
| 默认 `long_term_memory` 为 JSON object | 通过 |
| 默认 `facts` 为空数组 | 通过 |
| 默认 `summary` 为空字符串 | 通过 |
| 默认 `memory_version=0` | 通过 |
| 默认 `status=active` | 通过 |
| 重复通道身份被唯一约束拒绝 | 通过 |
| 非法 channel 被检查约束拒绝 | 通过 |
| 缺少 facts/summary 的记忆对象被拒绝 | 通过 |
| 验证数据全部回滚 | 通过，残留 0 行 |

### 10.4 应用切换与迁移记录

当前 `.env` 已切换为：

```dotenv
DATA_SERVICE_PROFILE=local
CONVERSATION_STORE=redis
PROFILE_STORE=postgres
CONVERSATION_STORAGE_REQUIRED=true
```

- 本地运行时先保持 SSH 隧道进程存活，读取 `POSTGRES_*` / `REDIS_*`。
- 测试服务器部署时使用相同代码，把 `DATA_SERVICE_PROFILE` 改为 `server`，读取 `POSTGRES_SERVER_*` / `REDIS_SERVER_*`。
- FastAPI 启动时检查两个后端；任一不可用会在启动阶段失败，不静默降级到本地内存。

迁移源 `state/cookclaw_conversation.db` 共 38 条历史记录。迁移时仅处理尚未过期的 19 条：

| 目标 | 源有效记录 | 写入结果 |
| --- | ---: | ---: |
| PostgreSQL 长期画像 | 14 | 14 |
| Redis 短期会话 | 5 | 5 |

迁移中发现旧版微信画像键缺少机器人账号，已兼容映射到 `channel_account_id=default`。再次执行迁移时 19 条全部跳过，确认“只写更高版本”的幂等保护有效。源 SQLite 文件未删除、未改写。

### 10.5 真实存储验证

可重复执行：

```bash
.venv/bin/python scripts/verify_conversation_storage.py
.venv/bin/python scripts/probe_conversation_storage.py
```

实际结果：

- Redis、PostgreSQL 健康检查均通过。
- 14 条 PG 画像和 5 条 Redis 会话逐条读取、版本核对通过。
- 应用写读探针通过：短期轮次可回读、同通道不同 thread 可共享 PG 画像、长期 fact 可追溯。
- 探针账号及 Redis Key 已自动清理；验证后仍为 14 条活跃画像、5 条短期会话。
- 会话与存储相关测试：65 项全部通过。
- 全仓测试：219 项通过、8 项失败；失败集中在此前产品链路改造后尚未更新的固定开场文案、澄清时机、召回数量和画像软偏好旧断言，不涉及本次 Redis/PG 存储读写。

### 10.6 回滚方式

发生应用兼容问题时，只改运行开关并重启：

```dotenv
CONVERSATION_STORE=sqlite
PROFILE_STORE=same
```

回滚时不要删除 PostgreSQL 表或 Redis Key；保留切换期间写入的数据，待问题修复后做增量合并。SQLite 在切换之后不会继续同步新写入，因此回滚只能用于紧急恢复，不能把它当作无损双写方案。

### 10.7 当前边界

- 已切换的是 `ConversationMemory` 短期会话和 `profile:*` 长期画像。
- 设备 pending/确认/未决执行态由 Redis 独立 task-state key 持久化；多 worker 故障注入仍待测试服务器验收。
- 不同通道身份仍相互隔离；只有同一通道、同一通道账号、同一用户标识共享长期画像。
- 本地与测试服务器可以读写同一套 PG/Redis，但同一通道账号不应同时消费同一批消息，以免重复回复。
