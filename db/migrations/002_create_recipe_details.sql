-- Mock Device 真实菜谱详情缓存
-- 一个 recipe_id 可按 Cook-Language 保存多份本地化详情。

BEGIN;

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

CREATE INDEX IF NOT EXISTS idx_recipe_details_fetched_at
    ON public.recipe_details (fetched_at DESC);

CREATE INDEX IF NOT EXISTS idx_recipe_details_name
    ON public.recipe_details (language, name);

COMMENT ON TABLE public.recipe_details IS
    'Mock Device 菜谱详情事实缓存：规范化详情用于用户展示，脱敏原始回执用于追溯';
COMMENT ON COLUMN public.recipe_details.detail IS
    'recipe_detail_v1：食材、步骤、媒体、时间、份量和设备参数摘要';
COMMENT ON COLUMN public.recipe_details.raw_payload IS
    '移除 isCollect/isPurchase 等调用账号状态后的远端 data 原文';

COMMIT;
