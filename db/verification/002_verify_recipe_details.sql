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
