"""受控闲聊规则测试（纯逻辑，不调用 LLM）。"""
from datetime import datetime
from zoneinfo import ZoneInfo

from app.orchestrator.smalltalk import (
    greeting_time_context,
    greeting_system_prompt,
    identity_system_prompt,
    is_greeting_message,
    is_identity_question,
    smalltalk_system_prompt,
)


def test_smalltalk_prompt_keeps_boundaries():
    prompt = smalltalk_system_prompt("zh")
    assert "不调用工具" in prompt
    assert "不启动或停止设备" in prompt
    assert "不编造真实菜谱库" in prompt
    assert "不要强行把话题拉回做饭" in prompt
    assert "实时数据" in prompt


def test_pure_greeting_detection_covers_tone_particles_without_swallowing_tasks():
    assert is_greeting_message("早上好")
    assert is_greeting_message("早上好呀～")
    assert is_greeting_message("哈喽啊")
    assert is_greeting_message("Good morning!")
    assert not is_greeting_message("早上好，推荐三个早餐")
    assert not is_greeting_message("你好，设备在线吗")


def test_greeting_prompt_avoids_recent_wording_and_fixed_cooking_redirect():
    prompt = greeting_system_prompt("zh", recent_replies=["早呀，今天想吃什么？"])
    assert "早呀，今天想吃什么？" in prompt
    assert "不得重复" in prompt
    assert "不要强行转回做饭" in prompt
    assert "连续问候" in prompt


def test_greeting_time_context_uses_configured_timezone_without_location_guessing():
    context = greeting_time_context(
        "早上好",
        timezone_name="Asia/Shanghai",
        now=datetime(2026, 7, 26, 16, 20, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert context["period"] == "afternoon"
    assert context["user_period"] == "morning"
    assert context["mismatch"] is True

    prompt = greeting_system_prompt("zh", time_context=context)
    assert "Asia/Shanghai" in prompt
    assert "当前时间段是“下午”" in prompt
    assert "不代表已知道用户所在地" in prompt
    assert "明确用“下午”回应" in prompt


def test_greeting_time_context_falls_back_for_invalid_timezone():
    context = greeting_time_context(
        "hello",
        timezone_name="Invalid/Timezone",
        now=datetime(2026, 7, 26, 2, 0, tzinfo=ZoneInfo("UTC")),
    )
    assert context["timezone"] == "Asia/Shanghai"
    assert context["period"] == "morning"
    assert context["mismatch"] is False


def test_identity_question_detection_covers_identity_and_capability():
    assert is_identity_question("你是谁？")
    assert is_identity_question("你能做什么")
    assert is_identity_question("Who are you?")
    assert not is_identity_question("西红柿炒鸡蛋怎么做")


def test_identity_prompt_separates_facts_from_expression():
    prompt = identity_system_prompt("zh", has_history=True)
    assert "真实菜谱库" in prompt
    assert "不得承诺永久记忆" in prompt
    assert "账号、权限" in prompt
    assert "不能逐条照抄" in prompt
    assert "聊了这么一会儿" in prompt


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)
