---
name: recipe-search
description: 基于混合检索（关键词 BM25 + 语义向量）的食谱搜索工具。阿里云百炼 Embedding + Milvus（dense/COSINE 与稀疏 BM25 双路、RRF 融合）+ 元数据过滤（语言/素食/忌口），实现智能食谱推荐。当用户提到"推荐食谱"、"我想做..."、"今天吃什么"、"有什么...的菜"、"食谱搜索"或任何与食谱检索、菜品推荐相关的需求时触发此 skill。
---

# Recipe Search Skill

基于**混合检索**的食谱搜索工具：语义向量（dense / COSINE）与关键词（BM25 稀疏）两路并行召回、RRF 融合，并按元数据硬过滤，从 Milvus 中返回最匹配的食谱。

## 检索能力（混合召回 + 元数据过滤 + 重排）

```
query → 槽位抽取(facets/素食/忌口) → 向量化(百炼 text-embedding-v4)
      → Milvus hybrid_search：dense(COSINE) + 关键词(BM25 稀疏)，RRF 融合
      → 元数据过滤(lang/素食) + 忌口后置过滤
      → gte-rerank-v2 精排(候选池 top-20 → top_k) → 结果
```

- **语义路 + 关键词路**：语义路召回"意思相近"，关键词路（jieba 分词 + BM25）召回"字面精确命中"，互补。确切菜名、低频食材（如"土豆丝""沙茶酱"）靠关键词路稳。
- **元数据硬过滤**（违反=事故才硬过滤）：
  - `lang`：按 query 语言过滤 zh/en，去跨语言重复
  - 素食：query 含"素食/纯素"时，按 `facets.diet` 硬过滤
  - 忌口：query 含"不要X/无X/忌X"时，剔除含该食材的菜
- 菜系/口味/烹饪方式等软维度已结构化进 `facets`，默认交给 hybrid 排序（不硬砍）。
- **重排（精排）**：召回候选池 top-20 交给 DashScope `gte-rerank-v2`（cross-encoder）精排到 top_k，显著提升前 3 命中（实测食材口语 query recall@3 84%→97%）。**重排失败自动回退召回原序，不阻断检索**；`RERANK_ENABLED=0` 可关。
- **查询理解**：槽位/忌口默认走规则版（`module/normalize.py`，已支持多词忌口拆分 `香菜和花生`/`香菜、葱和蒜`）；另有**可选 LLM 版**（`QU_ENABLED=1`，qwen-plus 把口语 query 理解成干净 search_query + 槽位，失败回退规则）。**实测 LLM 版对召回零增量，默认关**。
- 集合：线上查询用 `recipe_hybrid`；纯语义基线 `recipe_collection` 保留作评测/回滚。

## 执行流程

### 1. 提取搜索关键词

从用户输入中提取搜索条件，组合为简洁的查询文本。示例：

| 用户输入 | 提取的查询文本 |
|---------|--------------|
| "今天想吃清淡的鸡肉菜" | `清淡的鸡肉菜` |
| "推荐一道快手家常菜" | `快手家常菜` |
| "有什么用土豆和牛肉做的菜" | `土豆牛肉` |
| "我想做红烧肉" | `红烧肉` |
| "想吃素食，不要香菜" | `素食 不要香菜`（素食/忌口会被识别为硬过滤）|

### 2. 执行搜索命令

**必须**在技能目录下使用独立虚拟环境执行：

```bash
cd app/agent/skills/recipe-search && .venv/bin/python recipe_search.py "查询文本" top_k
```

**参数说明：**
- `查询文本`：从用户输入中提取的搜索关键词（必填，用引号包裹）
- `top_k`：返回结果数量（可选，默认 3，最大 20）

**执行示例：**

```bash
# 搜索清淡的鸡肉菜，返回 3 条结果
cd app/agent/skills/recipe-search && .venv/bin/python recipe_search.py "清淡的鸡肉菜" 3

# 搜索红烧肉
cd app/agent/skills/recipe-search && .venv/bin/python recipe_search.py "红烧肉" 3

# 带忌口的搜索
cd app/agent/skills/recipe-search && .venv/bin/python recipe_search.py "宫保鸡丁，不要花生" 3
```

> 应急回退：设环境变量 `HYBRID_ENABLED=0` 可临时退回纯语义检索。

### 3. 解析输出结果

**成功时**输出格式如下（文本模式不显示分数；结构化分数见 `--json`）：

```
🍳 为您找到以下食谱：

1. **炒土豆丝**
   🖼️ 图片：https://images.example.invalid/...
   🥘 食材：青椒、土豆、盐、蔬菜精、植物油
   🏷️ 标签：家常菜 / home-style / 清淡

2. **洋葱土豆丝**
   🖼️ 图片：https://images.example.invalid/...
   🥘 食材：洋葱、土豆、小葱、红椒、盐
   🏷️ 标签：家常菜 / home-style / 咸鲜
```

需要结构化数据（含相关度分数 `score`、`facets` 等）时加 `--json`：

```bash
.venv/bin/python recipe_search.py "查询文本" 3 --json
```

> 说明：`score` 是综合相关度——开启重排时为 `gte-rerank-v2` 的相关性分；未开启时为 RRF 融合分归一到 [0,1]（1.0≈两路都排第一）。均非纯余弦相似度。

**失败时**输出：

```
搜索失败：<错误信息>
```

### 4. 格式化回复用户

将搜索结果按以下格式回复用户，并追问是否需要执行某个食谱：

```
🍳 为您找到以下食谱：

1. **食谱名称**
   ![食谱图片](https://images.example.invalid/recipe.jpg)
   🥘 食材：xxx、xxx、xxx
   🏷️ 标签：清淡 / 家常 / 快手

2. **食谱名称**
   ![食谱图片](https://images.example.invalid/recipe.jpg)
   ...

💡 您想执行哪个食谱呢？告诉我食谱编号即可开始烹饪！
```

## 异常处理

| 异常情况 | 处理方式 |
|---------|---------|
| 命令执行报错 `ModuleNotFoundError` | 向用户报告："搜索服务暂时不可用，请稍后再试" |
| 输出 `搜索失败：...` | 如实告知用户搜索失败原因，建议换关键词重试 |
| 返回结果为空 | 告知用户未找到匹配食谱，建议更换搜索条件（过滤可能过严，如忌口太多）|
| 网络连接超时 | 告知用户网络异常，建议稍后重试 |

## 注意事项

- **禁止**凭空编造食谱数据，所有结果必须来自搜索命令的实际输出
- **禁止**使用项目根目录的 Python 环境执行，必须使用技能目录下的 `.venv/bin/python`
- 搜索结果中的食谱 ID 可用于触发 `recipe-operation` 技能执行烹饪
- 查询文本应简洁明确，避免过长的自然语言描述
- 混合库 `recipe_hybrid` 由 `migrate_hybrid.py` 从基线集合迁移生成（复用向量、零重新 embedding）；中文关键词召回依赖 `jieba` 分词（已在依赖中）
