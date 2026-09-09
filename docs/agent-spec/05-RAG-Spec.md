# 05. RAG Specification

## 1. 当前在线链路（AS-IS）

~~~mermaid
flowchart LR
    Q[标准化后的查询] --> SR[SearchRequest]
    SR --> PQ[检索查询规划]
    PQ --> EMB[DashScope text-embedding-v4\n1024 维]
    PQ --> BM[BM25 稀疏查询]
    EMB --> D[Milvus Dense COSINE]
    BM --> S[Milvus Sparse]
    D --> RRF[RRF 融合]
    S --> RRF
    RRF --> C[候选池]
    C --> RR[gte-rerank-v2]
    RR --> F[语言/忌口/质量过滤]
    F --> SEL[精确命中 + 多样性选择]
    SEL --> RESP[Grounded 格式化/受控推荐表达]
~~~

核心实现位于：

- `app/agent/fast_path.py`
- `app/agent/recipe_search_service.py`
- `app/agent/skills/recipe-search/recipe_search.py`
- `app/agent/skills/recipe-search/module/query_understanding.py`
- `app/agent/skills/recipe-search/module/embedding.py`
- `app/agent/skills/recipe-search/dataset/milvus.py`

CookClaw 当前不是“检索文本块后交给大模型自由生成答案”的传统 RAG。每条菜谱是一条结构化文档，搜索结果由代码按真实字段格式化。这一点显著降低了菜名、图片和食材幻觉。

## 2. 查询处理分层

| 层 | 当前作用 | 状态与风险 |
|---|---|---|
| 语义标准化 | 修正常见口误、识别候选引用 | 规则较窄，散落在编排层 |
| Intent 结构化 | 提取菜谱搜索意图和关键词 | 依赖 `qwen-plus`，失败会降到 unknown |
| SearchRequest | 合并当前槽位、排除项和弱记忆 | 已是搜索领域契约；内部解析仍较大，后续可继续拆分 |
| Search Query Plan | 针对少数场景扩展查询 | 目前只覆盖部分减脂、辣味、饮酒后等场景 |
| RAG Query Understanding | 生成 facets/可选 LLM QU | `QU_ENABLED=0` 默认关闭，以规则为主 |

当前各层已经通过 `SearchRequest` 和路由结构关联；仍需继续收敛语义改写的输入、输出和
reason code，避免“麻烦”被拆成“麻味”等错误。

## 3. 召回与排序

### 3.1 Dense

- 模型：DashScope `text-embedding-v4`。
- 维度：1024。
- 距离：COSINE。
- 用途：召回语义相近但字面不同的菜谱。

### 3.2 Sparse/BM25

- Milvus 稀疏字段配合 BM25。
- 用途：保护菜名、主料、菜系等精确词命中。
- Milvus Lite 使用 `SPARSE_INVERTED_INDEX`，混合集合灌库应保持单次 insert，避免已知的跨 segment 问题。

### 3.3 融合与重排

- Dense 与 Sparse 使用 RRF 融合。
- 当前召回参数以 `recall_k=30`、`rrf_k=60` 为基线。
- 展示前候选池通常为 20 条。
- 默认使用 `gte-rerank-v2` 重排；失败时 SHOULD 保留融合结果顺序并标记降级。

### 3.4 过滤与选择

当前结果还会经过：

1. 语言、素食等元数据过滤；
2. 图片、食材等最低质量过滤；
3. 本轮禁忌和排除项的二次硬过滤；
4. 明确菜名的精确命中优先；
5. 菜名、做法和口味层面的去重与多样性选择；
6. 结果数量裁剪，最多展示 10 道。

安全型过滤结果为 0 时 MUST 返回空结果，不能为了“有答案”恢复被排除菜谱。

## 4. Prompt 与上下文

在线搜索先由真实检索和硬过滤得到候选，再把裁剪后的候选字段交给可选推荐表达模型。
大模型主要用于：

- 意图分类；
- 可选 Query Understanding；
- 在“当前三道里选一道”等受限任务中，只选择真实候选；
- 在 grounded 推荐链中根据传入字段组织自然理由，输出还要经过事实校验。

推荐理由应由菜谱真实字段构造。若数据没有热量、时间或营养字段，就不能宣称“最低卡”“只要十分钟”或“最适合糖尿病患者”。健康目标只能基于可见食材、做法、口味做保守判断，并说明证据边界。

## 5. Chunk、TopK 与 Context 的准确解释

- **Chunk**：当前没有文章式切块；一条菜谱即一条检索文档。
- **Recall TopK**：召回层保留较大候选池，为重排和过滤服务。
- **Display TopK**：由用户数量要求和会话场景决定，默认 3 道。
- **Context**：搜索上下文来自当前 SearchRequest、候选快照和受控弱记忆，不是把完整聊天历史无差别拼入 query。
- **Reply**：卡片数据由确定性格式器生成；推荐叙述可由受控模型生成，但只能引用提供的
  真实字段并经过校验，不允许补造数据库缺失字段。

## 6. 当前容易降低召回的地方

| 风险 | 影响 | 建议 |
|---|---|---|
| Query Rewrite 分散在多层 | 约束丢失、重复扩写或相互覆盖 | 收敛为一个可审计请求对象 |
| 语言检测主要区分中/英 | 其它语言容易被归为英文 | 引入 locale 与置信度 |
| 重排文档偏中文标签 | 英文候选的判别信息不足 | 按语言构造等价 rerank 文本 |
| 离线标签质量不稳定 | 错标签会同时污染过滤、排序和理由 | 增加标签来源、置信度和抽检 |
| 最低相似度配置未真正生效 | 弱相关结果仍可能进入展示 | 接入阈值，并按 query 类型校准 |
| 硬过滤空集后恢复原结果 | 可能违反用户排除条件 | 安全条件禁止回填 |
| 真实用户评测样本不足 | 离线高分无法代表连续对话 | 用匿名化内测日志建立回归集 |

## 7. 当前容易答非所问或产生幻觉的地方

1. 候选追问若未先识别上下文动作，会被重新分类成一次搜索；当前已有前置候选规则，但需统一覆盖所有表达。
2. “减肥、控糖、低脂”等目标缺少结构化营养字段，模型容易过度推断。
3. 历史偏好若被直接拼进当前 query，会把“想喝汤”错误收窄成“牛肉汤”。
4. 离线生成标签若不准确，确定性文案仍会把错误标签说得很肯定。
5. 通用 cooking QA 当前不查菜谱库，不应回答具体菜谱数据库事实。

## 8. 目标 RAG 契约（TO-BE）

每次检索 SHOULD 产生以下可观测信息：

~~~json
{
  "trace_id": "...",
  "raw_query": "...",
  "normalized_query": "...",
  "search_request": {},
  "memory_fields_used": [],
  "filters": {},
  "dense_hits": 30,
  "sparse_hits": 30,
  "rerank_degraded": false,
  "hard_filter_dropped": 2,
  "displayed_recipe_ids": [],
  "latency_ms": {}
}
~~~

该记录不得保存不必要的原始个人信息，日志分析使用脱敏 query 或经授权的内测数据。

## 9. 评测规范

至少分开评测：

- 精确菜名命中率；
- 食材/菜系/口味/做法约束满足率；
- 忌口违规率，目标必须为 0；
- Recall@3、MRR、NDCG；
- 连续追问保持当前候选的准确率；
- “换一批”的非重复率；
- 中英文等价查询一致性；
- 空结果诚实率和工具失败不编造率；
- 搜索端到端 P50/P95 延迟。

现有约 98.7% 的 recall@3 可作为工程基线，但不能替代真实内测会话集。新增匿名回归集后，应按意图、语言、通道和是否有上下文分桶报告。
