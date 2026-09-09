"""高风险拒判场景的统一、自然且无副作用文案。"""
from __future__ import annotations


def affirmative_without_pending_message(
    lang: str,
    *,
    has_recent_context: bool = False,
) -> str:
    """裸肯定词没有可绑定状态时，只澄清，不暗示已经执行。"""
    if lang == "en":
        return (
            "Sure — I don't have a step waiting for confirmation right now. Which part of our earlier conversation would you like to continue?"
            if has_recent_context
            else "Sure — I don't have a step waiting for confirmation right now. What would you like to do?"
        )
    return (
        "可以，不过我这边现在没有正在等确认的步骤。你想接着刚才哪一点？"
        if has_recent_context
        else "可以，不过我这边现在没有正在等确认的步骤。你想先做什么？"
    )
