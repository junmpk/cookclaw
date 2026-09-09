# PostgreSQL 菜谱详情表结构与维护

> **更新日期**: 2026-08-07  
> **数据库**: PostgreSQL  
> **表名**: `public.recipe_details`

---

## 📊 表结构概览

### 表定义

```sql
CREATE TABLE IF NOT EXISTS public.recipe_details (
    recipe_id VARCHAR(80) NOT NULL,
    language VARCHAR(16) NOT NULL,
    name VARCHAR(255) NOT NULL DEFAULT '',

    detail JSONB NOT NULL,
    raw_payload JSONB NOT NULL,

    source_updated_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_recipe_details PRIMARY KEY (recipe_id, language),
    CONSTRAINT ck_recipe_details_language
        CHECK (language IN ('zh', 'en')),
    CONSTRAINT ck_recipe_details_detail_object
        CHECK (jsonb_typeof(detail) = 'object'),
    CONSTRAINT ck_recipe_details_raw_object
        CHECK (jsonb_typeof(raw_payload) = 'object'),
    CONSTRAINT ck_recipe_details_schema
        CHECK (detail ->> 'schema_version' = 'recipe_detail_v1'),
    CONSTRAINT ck_recipe_details_id_match
        CHECK (detail ->> 'recipe_id' = recipe_id)
);
```

### 字段说明

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `recipe_id` | VARCHAR(80) | 菜谱 ID（Mock Device 的 cookId） | 主键之一 |
| `language` | VARCHAR(16) | 语言（zh 或 en） | 主键之一，必须是 'zh' 或 'en' |
| `name` | VARCHAR(255) | 菜谱名称 | 默认空字符串 |
| `detail` | JSONB | 规范化的菜谱详情（recipe_detail_v1 schema） | 必须是对象，schema_version 必须是 'recipe_detail_v1' |
| `raw_payload` | JSONB | 原始 API 响应（脱敏后） | 必须是对象 |
| `source_updated_at` | TIMESTAMPTZ | Mock Device 源数据的更新时间 | 可为 NULL |
| `fetched_at` | TIMESTAMPTZ | 从 Mock Device 抓取的时间 | 默认 NOW() |
| `created_at` | TIMESTAMPTZ | 记录创建时间 | 默认 NOW() |
| `updated_at` | TIMESTAMPTZ | 记录更新时间 | 默认 NOW() |

### 索引

| 索引名 | 字段 | 说明 |
|---|---|---|
| `pk_recipe_details` | (recipe_id, language) | 主键索引 |
| `idx_recipe_details_fetched_at` | fetched_at DESC | 按抓取时间排序 |
| `idx_recipe_details_name` | (language, name) | 按语言和名称搜索 |

---

## 📋 detail 字段结构（recipe_detail_v1）

`detail` 字段存储规范化的菜谱详情，遵循 `recipe_detail_v1` schema。

### 完整结构

```json
{
  "schema_version": "recipe_detail_v1",
  "recipe_id": "12345",
  "cookId": "12345",
  "language": "zh",
  "name": "番茄炒蛋",
  "introduction": "经典家常菜，简单快手",
  "tips": "鸡蛋要先打散，番茄要切小块",
  "media": {
    "landscape_image_url": "https://example.com/image1.jpg",
    "portrait_image_url": "https://example.com/image2.jpg",
    "intro_video_url": "https://example.com/video.mp4"
  },
  "tags": ["家常菜", "快手菜", "下饭"],
  "ingredients": [
    {
      "group": "main",
      "name": "番茄",
      "amount": 2,
      "unit": "个",
      "remark": "中等大小"
    },
    {
      "group": "main",
      "name": "鸡蛋",
      "amount": 3,
      "unit": "个",
      "remark": null
    },
    {
      "group": "seasoning",
      "name": "盐",
      "amount": 5,
      "unit": "克",
      "remark": null
    }
  ],
  "steps": [
    {
      "number": 1,
      "type": "manual",
      "description": "番茄洗净切小块，鸡蛋打散加少许盐搅匀",
      "image_url": null,
      "video_url": null,
      "duration_seconds": 120,
      "parameters": []
    },
    {
      "number": 2,
      "type": "device",
      "description": "热锅凉油，倒入鸡蛋液炒至凝固盛出",
      "image_url": null,
      "video_url": null,
      "duration_seconds": 60,
      "parameters": [
        {
          "parameter_id": "step2_param",
          "time_seconds": 60,
          "temperature_c": 180,
          "speed": "中火",
          "power": null,
          "turn": null,
          "weight": null,
          "preset_pressure": null,
          "cook_pressure": null,
          "accessory_type": null,
          "accessory_image_url": null,
          "device_models": ["model_a", "model_b"]
        }
      ]
    }
  ],
  "cooking_time_seconds": 600,
  "servings": 2,
  "challenge_level": 1,
  "calorie_number": 250.5,
  "category_ids": ["cat_001", "cat_002"],
  "accessory_ids": [],
  "device_model_ids": ["model_a", "model_b"],
  "is_custom_food": false,
  "executable": true,
  "source_created_at": "2024-01-01T10:00:00Z",
  "source_updated_at": "2024-06-01T15:30:00Z"
}
```

### 字段详解

#### 基本信息

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | string | 固定为 "recipe_detail_v1" |
| `recipe_id` | string | 菜谱 ID |
| `cookId` | string | 菜谱 ID（与 recipe_id 相同） |
| `language` | string | 语言（zh 或 en） |
| `name` | string | 菜谱名称 |
| `introduction` | string \| null | 菜谱简介 |
| `tips` | string \| null | 烹饪小技巧 |

#### 媒体信息（media）

| 字段 | 类型 | 说明 |
|---|---|---|
| `landscape_image_url` | string \| null | 横版图片 URL |
| `portrait_image_url` | string \| null | 竖版图片 URL |
| `intro_video_url` | string \| null | 介绍视频 URL |

#### 标签（tags）

- 类型：`string[]`
- 说明：菜谱标签列表，如 ["家常菜", "快手菜"]

#### 食材（ingredients）

- 类型：`Ingredient[]`

```typescript
interface Ingredient {
  group: "main" | "accessory" | "seasoning" | "other";
  name: string;
  amount: number | null;
  unit: string | null;
  remark: string | null;
}
```

**字段说明**：
- `group`: 食材分组（主料、辅料、调料、其他）
- `name`: 食材名称
- `amount`: 用量（可为 null）
- `unit`: 单位（如 "克"、"个"、"毫升"）
- `remark`: 备注（如 "切小块"、"中等大小"）

#### 烹饪步骤（steps）

- 类型：`Step[]`

```typescript
interface Step {
  number: number;
  type: "manual" | "device";
  description: string;
  image_url: string | null;
  video_url: string | null;
  duration_seconds: number | null;
  parameters: StepParameter[];
}

interface StepParameter {
  parameter_id: string | null;
  time_seconds: number | null;
  temperature_c: number | null;
  speed: string | null;
  power: number | null;
  turn: number | null;
  weight: number | null;
  preset_pressure: number | null;
  cook_pressure: number | null;
  accessory_type: number | null;
  accessory_image_url: string | null;
  device_models: string[];
}
```

**字段说明**：
- `number`: 步骤序号
- `type`: 步骤类型（manual=手动步骤，device=设备步骤）
- `description`: 步骤描述
- `image_url`: 步骤图片 URL
- `video_url`: 步骤视频 URL
- `duration_seconds`: 步骤耗时（秒）
- `parameters`: 设备参数（仅 device 类型步骤有）

#### 其他信息

| 字段 | 类型 | 说明 |
|---|---|---|
| `cooking_time_seconds` | number \| null | 总烹饪时间（秒） |
| `servings` | number \| null | 份数 |
| `challenge_level` | number \| null | 难度等级（1-5） |
| `calorie_number` | number \| null | 卡路里数 |
| `category_ids` | string[] | 分类 ID 列表 |
| `accessory_ids` | string[] | 配件 ID 列表 |
| `device_model_ids` | string[] | 设备型号 ID 列表 |
| `is_custom_food` | boolean | 是否为自定义菜谱 |
| `executable` | boolean | 是否可由设备执行 |
| `source_created_at` | string \| null | 源数据创建时间（ISO 8601） |
| `source_updated_at` | string \| null | 源数据更新时间（ISO 8601） |

---

## 📋 raw_payload 字段结构

`raw_payload` 存储从 Mock Device Adapter 获取的原始响应，但移除了敏感字段。

### 移除的敏感字段

```python
_PRIVATE_RESPONSE_FIELDS = {
    "isCollect",      # 用户是否收藏
    "isPurchase",     # 用户是否购买
}
```

### 原始响应示例

```json
{
  "id": "12345",
  "name": "番茄炒蛋",
  "recipeIntroduce": "经典家常菜",
  "recipeTips": "鸡蛋要先打散",
  "landscapeImageUrl": "https://example.com/image1.jpg",
  "portraitImageUrl": "https://example.com/image2.jpg",
  "introduceVideoUrl": "https://example.com/video.mp4",
  "tagVoList": [
    {"name": "家常菜"},
    {"name": "快手菜"}
  ],
  "recipeIngredientsVoList": {
    "mainMaterials": [
      {"foodIngredientName": "番茄", "amount": 2, "unitName": "个"}
    ],
    "recipeSeasoning": [
      {"foodIngredientName": "盐", "amount": 5, "unitName": "克"}
    ]
  },
  "recipeStepVoList": [
    {
      "serialNumb": 1,
      "stepDesc": "番茄洗净切小块",
      "stepTime": 120
    }
  ],
  "cookingTime": 600,
  "serviceSize": 2,
  "challengeLevel": 1,
  "calorieNumber": 250.5,
  "createTime": 1704067200000,
  "updateTime": 1717255800000
}
```

---

## 🔧 表结构维护

### 迁移文件位置

```
db/migrations/
├── 001_create_channel_users.sql
└── 002_create_recipe_details.sql    ← 菜谱详情表
```

### 验证文件位置

```
db/verification/
├── 001_verify_channel_users.sql
└── 002_verify_recipe_details.sql    ← 菜谱详情表验证
```

### 应用代码位置

```
app/
├── recipe_detail_store.py          ← PostgreSQL 存储层
└── orchestrator/
    └── recipe_detail.py            ← Schema 规范化与格式化
```

---

## 🚀 数据库操作

### 1. 运行迁移

```bash
# 连接到 PostgreSQL
psql -h <host> -U <user> -d <database>

# 运行迁移
\i db/migrations/002_create_recipe_details.sql
```

### 2. 验证表结构

```bash
# 运行验证脚本
psql -h <host> -U <user> -d <database> -f db/verification/002_verify_recipe_details.sql
```

**预期输出**：
```
 table_exists | detail_count | invalid_count | private_state_count
--------------+--------------+---------------+---------------------
 t            |         1234 |             0 |                   0
```

### 3. 查询菜谱详情

```sql
-- 查询指定菜谱
SELECT recipe_id, language, name, detail
FROM public.recipe_details
WHERE recipe_id = '12345' AND language = 'zh';

-- 查询最近抓取的菜谱
SELECT recipe_id, name, fetched_at
FROM public.recipe_details
ORDER BY fetched_at DESC
LIMIT 10;

-- 按名称搜索
SELECT recipe_id, name
FROM public.recipe_details
WHERE language = 'zh' AND name ILIKE '%番茄%'
ORDER BY name;
```

### 4. 插入/更新菜谱详情

```sql
-- 插入或更新（UPSERT）
INSERT INTO public.recipe_details (
    recipe_id,
    language,
    name,
    detail,
    raw_payload,
    source_updated_at,
    fetched_at,
    updated_at
) VALUES (
    '12345',
    'zh',
    '番茄炒蛋',
    '{"schema_version": "recipe_detail_v1", ...}'::jsonb,
    '{"id": "12345", ...}'::jsonb,
    '2024-06-01T15:30:00Z',
    NOW(),
    NOW()
)
ON CONFLICT (recipe_id, language)
DO UPDATE SET
    name = EXCLUDED.name,
    detail = EXCLUDED.detail,
    raw_payload = EXCLUDED.raw_payload,
    source_updated_at = EXCLUDED.source_updated_at,
    fetched_at = NOW(),
    updated_at = NOW();
```

### 5. 删除菜谱详情

```sql
-- 删除指定菜谱
DELETE FROM public.recipe_details
WHERE recipe_id = '12345' AND language = 'zh';

-- 删除过期的菜谱（超过 30 天未更新）
DELETE FROM public.recipe_details
WHERE fetched_at < NOW() - INTERVAL '30 days';
```

---

## 📝 数据写入流程

### 1. 从 Mock Device Adapter 抓取

```python
from app.orchestrator.recipe_detail import normalize_recipe_detail, sanitize_recipe_detail_payload
from app.recipe_detail_store import save_recipe_detail

# 从 Mock Device Adapter 获取原始数据
raw_response = await kitchen_idea_api.get_recipe(recipe_id)

# 规范化为 recipe_detail_v1
detail = normalize_recipe_detail(raw_response, language="zh")

# 脱敏原始响应
raw_payload = sanitize_recipe_detail_payload(raw_response)

# 保存到 PostgreSQL
await save_recipe_detail(detail, raw_payload)
```

### 2. 从 PostgreSQL 读取

```python
from app.recipe_detail_store import load_recipe_detail

# 查询菜谱详情
detail = await load_recipe_detail(recipe_id="12345", language="zh")

if detail is None:
    print("菜谱不存在")
else:
    print(detail["name"])
    print(detail["ingredients"])
    print(detail["steps"])
```

---

## 🔍 数据验证

### 验证规则

1. **主键约束**：(recipe_id, language) 必须唯一
2. **语言约束**：language 必须是 'zh' 或 'en'
3. **JSON 类型约束**：detail 和 raw_payload 必须是 JSON 对象
4. **Schema 版本约束**：detail->>'schema_version' 必须是 'recipe_detail_v1'
5. **ID 匹配约束**：detail->>'recipe_id' 必须等于 recipe_id 字段

### 验证脚本

```sql
-- 验证所有记录
SELECT
    to_regclass('public.recipe_details') IS NOT NULL AS table_exists,
    COUNT(*) AS detail_count,
    COUNT(*) FILTER (
        WHERE detail ->> 'schema_version' <> 'recipe_detail_v1'
           OR detail ->> 'recipe_id' <> recipe_id
    ) AS invalid_count,
    COUNT(*) FILTER (
        WHERE raw_payload ? 'isCollect'
           OR raw_payload ? 'isPurchase'
    ) AS private_state_count
FROM public.recipe_details;
```

**预期结果**：
- `table_exists`: true
- `invalid_count`: 0（所有记录都符合 schema）
- `private_state_count`: 0（没有敏感字段）

---

## 📊 性能优化

### 索引使用

```sql
-- 按 recipe_id + language 查询（使用主键索引）
SELECT * FROM recipe_details WHERE recipe_id = '12345' AND language = 'zh';

-- 按抓取时间排序（使用 idx_recipe_details_fetched_at）
SELECT * FROM recipe_details ORDER BY fetched_at DESC LIMIT 10;

-- 按语言 + 名称搜索（使用 idx_recipe_details_name）
SELECT * FROM recipe_details WHERE language = 'zh' AND name = '番茄炒蛋';
```

### JSONB 查询优化

```sql
-- 查询包含特定标签的菜谱
SELECT * FROM recipe_details
WHERE detail @> '{"tags": ["家常菜"]}'::jsonb;

-- 查询难度等级为 1 的菜谱
SELECT * FROM recipe_details
WHERE (detail ->> 'challenge_level')::int = 1;

-- 查询烹饪时间小于 30 分钟的菜谱
SELECT * FROM recipe_details
WHERE (detail ->> 'cooking_time_seconds')::int < 1800;
```

**注意**：如果需要频繁查询 JSONB 字段，建议创建 GIN 索引：

```sql
CREATE INDEX idx_recipe_details_detail_gin
    ON public.recipe_details USING GIN (detail);
```

---

## 🆘 常见问题

### Q1: 如何批量导入菜谱？

**A**: 使用 Python 脚本批量导入：

```python
import asyncio
from app.recipe_detail_store import save_recipe_detail
from app.orchestrator.recipe_detail import normalize_recipe_detail

async def bulk_import(recipes: list[dict], language: str = "zh"):
    for raw_recipe in recipes:
        detail = normalize_recipe_detail(raw_recipe, language)
        raw_payload = sanitize_recipe_detail_payload(raw_recipe)
        await save_recipe_detail(detail, raw_payload)
        print(f"Imported: {detail['name']}")

# 运行
asyncio.run(bulk_import(recipes_data, "zh"))
```

### Q2: 如何清理过期数据？

**A**: 定期清理超过 N 天未更新的记录：

```sql
-- 清理 90 天未更新的记录
DELETE FROM public.recipe_details
WHERE fetched_at < NOW() - INTERVAL '90 days';

-- 查看清理前的记录数
SELECT COUNT(*) FROM public.recipe_details
WHERE fetched_at < NOW() - INTERVAL '90 days';
```

### Q3: 如何修复损坏的数据？

**A**: 找出并修复不符合 schema 的记录：

```sql
-- 找出无效记录
SELECT recipe_id, language, detail
FROM public.recipe_details
WHERE detail ->> 'schema_version' <> 'recipe_detail_v1'
   OR detail ->> 'recipe_id' <> recipe_id;

-- 删除无效记录
DELETE FROM public.recipe_details
WHERE detail ->> 'schema_version' <> 'recipe_detail_v1'
   OR detail ->> 'recipe_id' <> recipe_id;
```

### Q4: 如何添加新的字段？

**A**: 
1. 更新 `app/orchestrator/recipe_detail.py` 中的 `normalize_recipe_detail()` 函数
2. 更新 `recipe_detail_v1` schema 文档
3. 运行迁移脚本（如果需要修改表结构）
4. 重新导入数据

---

## 📚 相关文档

- [P0 用户与记忆存储方案](./CookClaw-P0用户与记忆存储方案.md)
- [菜谱详情存储代码](../../app/recipe_detail_store.py)
- [菜谱详情规范化代码](../../app/orchestrator/recipe_detail.py)

---

**文档版本**: 1.0  
**更新日期**: 2026-08-07  
**维护团队**: CookClaw 架构组
