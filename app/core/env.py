"""Environment variable helpers shared by runtime data stores."""

from __future__ import annotations

import os
import re

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def getenv_resolved(name: str, default: str | None = None) -> str | None:
    """Read an env value and resolve exact ``${OTHER_ENV}`` references.

    ``python-dotenv`` expands these references when it owns the load. Some
    process launchers inject ``.env`` values first, though, leaving the raw
    reference in ``os.environ``; ``load_dotenv()`` then keeps that inherited
    value. Resolve that narrow form here without overriding normal environment
    precedence.
    """
    value = os.getenv(name)
    if value is None:
        return default

    visited = {name}
    while match := _ENV_REFERENCE.fullmatch(value.strip()):
        referenced_name = match.group(1)
        if referenced_name in visited:
            raise ValueError(
                f"cyclic environment variable reference at {referenced_name}"
            )
        visited.add(referenced_name)
        referenced_value = os.getenv(referenced_name)
        if referenced_value is None:
            raise ValueError(
                f"{name} references unset environment variable {referenced_name}"
            )
        value = referenced_value
    return value
