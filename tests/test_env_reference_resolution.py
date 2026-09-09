import asyncio

import pytest

from app.conversation.postgres_profile_store import PostgresChannelUserProfileStore
from app.conversation.redis_store import RedisConversationStore
from app.core.env import getenv_resolved
from app.recipe_detail_store import PostgresRecipeDetailStore


def test_getenv_resolved_follows_exact_reference(monkeypatch):
    monkeypatch.setenv("TEST_BASE_VALUE", "resolved")
    monkeypatch.setenv("TEST_ALIAS_VALUE", "${TEST_BASE_VALUE}")

    assert getenv_resolved("TEST_ALIAS_VALUE") == "resolved"


def test_getenv_resolved_rejects_unset_reference(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_VALUE", raising=False)
    monkeypatch.setenv("TEST_ALIAS_VALUE", "${TEST_MISSING_VALUE}")

    with pytest.raises(ValueError, match="references unset environment variable"):
        getenv_resolved("TEST_ALIAS_VALUE")


def _set_server_aliases(monkeypatch):
    monkeypatch.setenv("DATA_SERVICE_PROFILE", "server")

    monkeypatch.setenv("REDIS_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_SERVER_PORT", "6379")
    monkeypatch.setenv("REDIS_SERVER_PASSWORD", "${REDIS_PASSWORD}")
    monkeypatch.setenv("REDIS_SERVER_DB", "${REDIS_DB}")
    monkeypatch.setenv("REDIS_SERVER_SSL", "${REDIS_SSL}")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_SSL", "true")

    monkeypatch.setenv("POSTGRES_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_SERVER_PORT", "5432")
    monkeypatch.setenv("POSTGRES_SERVER_USER", "${POSTGRES_USER}")
    monkeypatch.setenv("POSTGRES_SERVER_PASSWORD", "${POSTGRES_PASSWORD}")
    monkeypatch.setenv("POSTGRES_SERVER_DB", "${POSTGRES_DB}")
    monkeypatch.setenv("POSTGRES_SERVER_SSLMODE", "${POSTGRES_SSLMODE}")
    monkeypatch.setenv("POSTGRES_USER", "cookclaw")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres-test-password")
    monkeypatch.setenv("POSTGRES_DB", "cookclaw")
    monkeypatch.setenv("POSTGRES_SSLMODE", "require")


def test_redis_server_config_resolves_inherited_references(monkeypatch):
    _set_server_aliases(monkeypatch)

    store = RedisConversationStore.from_env()
    kwargs = store.client.connection_pool.connection_kwargs

    assert kwargs["password"] == "redis-test-password"
    assert kwargs["db"] == 2
    assert kwargs["ssl_cert_reqs"] == "required"
    asyncio.run(store.close())


@pytest.mark.parametrize(
    "store_type",
    [PostgresChannelUserProfileStore, PostgresRecipeDetailStore],
)
def test_postgres_server_config_resolves_inherited_references(monkeypatch, store_type):
    _set_server_aliases(monkeypatch)

    store = store_type.from_env()

    assert store._connect_kwargs["user"] == "cookclaw"
    assert store._connect_kwargs["password"] == "postgres-test-password"
    assert store._connect_kwargs["database"] == "cookclaw"
    assert store._connect_kwargs["ssl"] == "require"
