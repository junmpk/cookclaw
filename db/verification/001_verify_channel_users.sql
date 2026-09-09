\set ON_ERROR_STOP on

-- 所有写入均在事务中并最终回滚，验证后不保留测试用户。
BEGIN;

INSERT INTO public.channel_users (
    channel,
    channel_account_id,
    channel_user_id,
    display_name
) VALUES (
    'qq',
    'migration_probe',
    'migration_probe_user',
    'Migration Probe'
);

DO $verify$
BEGIN
    BEGIN
        INSERT INTO public.channel_users (
            channel,
            channel_account_id,
            channel_user_id
        ) VALUES (
            'qq',
            'migration_probe',
            'migration_probe_user'
        );
        RAISE EXCEPTION 'duplicate identity was unexpectedly accepted';
    EXCEPTION
        WHEN unique_violation THEN
            RAISE NOTICE 'unique identity constraint: passed';
    END;

    BEGIN
        INSERT INTO public.channel_users (
            channel,
            channel_user_id
        ) VALUES (
            'invalid_channel',
            'migration_probe_invalid_channel'
        );
        RAISE EXCEPTION 'invalid channel was unexpectedly accepted';
    EXCEPTION
        WHEN check_violation THEN
            RAISE NOTICE 'channel check constraint: passed';
    END;

    BEGIN
        INSERT INTO public.channel_users (
            channel,
            channel_user_id,
            long_term_memory
        ) VALUES (
            'web',
            'migration_probe_invalid_memory',
            '{}'::jsonb
        );
        RAISE EXCEPTION 'invalid memory payload was unexpectedly accepted';
    EXCEPTION
        WHEN check_violation THEN
            RAISE NOTICE 'memory shape constraints: passed';
    END;
END
$verify$;

SELECT
    channel,
    channel_account_id,
    jsonb_typeof(profile) AS profile_type,
    jsonb_typeof(long_term_memory) AS memory_type,
    jsonb_array_length(long_term_memory -> 'facts') AS fact_count,
    long_term_memory ->> 'summary' AS summary,
    memory_version,
    status
FROM public.channel_users
WHERE channel = 'qq'
  AND channel_account_id = 'migration_probe'
  AND channel_user_id = 'migration_probe_user';

ROLLBACK;

SELECT COUNT(*) AS remaining_probe_rows
FROM public.channel_users
WHERE channel_account_id = 'migration_probe'
   OR channel_user_id LIKE 'migration_probe%';
