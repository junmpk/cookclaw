"""Web unified turn 的聚餐澄清状态交接回归。"""

from __future__ import annotations

import asyncio

from app.conversation.models import ConversationTaskState
from app.conversation.service import ConversationService
from app.conversation.store import InMemoryConversationStore
from app.conversation.task_state_repository import RedisDialogueTaskStateRepository
from app.orchestrator import router
from app.orchestrator.intent import IntentResult
from app.orchestrator.recommendation_response import RecommendationNarrative
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.turn.application_service import TurnApplicationService
from app.orchestrator.turn.runtime_models import ResponseEnvelope, TurnRequest
from app.ports import dialogue_state

_PARTY_REQUEST = (
    "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，要喝点白酒，"
    "所有人都没什么忌口，请推荐一个食谱清单给我"
)


def _service(store: InMemoryConversationStore) -> ConversationService:
    return ConversationService(store, profile_store=InMemoryConversationStore())


def test_same_web_thread_repeated_complete_request_recommends_immediately(
    monkeypatch,
):
    """完整聚餐条件每一轮都直推，不追问软偏好或产生待澄清状态。"""

    menu_plan_requests: list[router.SearchRequest] = []

    async def classify(question: str, routing_context=None) -> IntentResult:
        assert question == _PARTY_REQUEST
        return IntentResult.from_raw(
            {
                "r": True,
                "c": "recipe_recommend",
                "k": ["江西", "白酒", "朋友聚餐"],
                "s": {
                    "q": "杭州 白酒 赣菜 辣",
                    "dish": [],
                    "ingredient": [],
                    # 模拟分类模型把所在地/籍贯错误扩写成菜系和辣味；
                    # SearchRequest 必须在进入状态机前确定性清掉。
                    "cuisine": ["赣菜"],
                    "flavor": ["辣"],
                    "method": [],
                    "scene": ["朋友聚餐"],
                    "meal": ["晚餐"],
                    "diet": [],
                    "exclude": [],
                    "avoid": [],
                },
                "signals": {"no_constraints": True},
            },
            original_text=question,
            routing_context=routing_context,
        )

    async def menu_plan(request, *, lang: str, on_search_start=None) -> dict:
        del lang, on_search_start
        menu_plan_requests.append(request)
        return {
            "success": True,
            "results": [
                {
                    "id": "party-1",
                    "score": 0.9,
                    "metadata": {
                        "recipe_id": "party-1",
                        "name": "已核验聚餐菜",
                        "ingredients": ["鸡肉"],
                        "tags": ["家常"],
                    },
                }
            ],
            "_menu_plan": {"queries": [{"query": "下酒聚餐"}]},
        }

    async def narrative(**_kwargs) -> RecommendationNarrative:
        return RecommendationNarrative("", "", {}, "")

    async def clarification(*_args, dimension=None, **_kwargs) -> str:
        return f"clarify:{dimension}"

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)
    monkeypatch.setattr(router, "generate_recommendation_narrative", narrative)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarification)

    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        application = TurnApplicationService(task_state_repository=repository)
        thread_id = "web:repeat-party-constraints"
        observed: list[tuple[str | None, bool, str, str | None, list[str], list[str]]] = []
        persisted_pending: list[dict | None] = []

        async def runtime_loader(_context):
            raise AssertionError("最小路由回归不应加载额外运行时")

        async def recipe_handler(context) -> ResponseEnvelope:
            pending = dialogue_state.get_search_clarification(thread_id)
            outcome = await router.route_fast_path(
                context.request.utterance,
                pending_search_request=(pending or {}).get("request"),
                pending_clarification_dimension=(pending or {}).get("dimension"),
                pending_asked_dimensions=list(
                    (pending or {}).get("asked_dimensions") or []
                ),
                routing_context=RoutingContext(
                    pending_clarification=pending,
                    current_task="recipe_search" if pending else None,
                    channel="web",
                ),
            )
            assert outcome.search_request is not None
            observed.append(
                (
                    (pending or {}).get("dimension"),
                    outcome.search_request.constraints_confirmed,
                    outcome.kind,
                    outcome.clarification_dimension,
                    outcome.search_request.cuisines,
                    outcome.search_request.flavors,
                )
            )
            if outcome.kind == "clarify":
                dimension = str(outcome.clarification_dimension or "")
                asked = [
                    *(pending or {}).get("asked_dimensions", []),
                    dimension,
                ]
                dialogue_state.set_search_clarification(
                    thread_id,
                    outcome.search_request.model_dump(),
                    dimension=dimension,
                    asked_dimensions=asked,
                    lang=outcome.lang,
                )
            else:
                dialogue_state.clear_search_clarification(thread_id)
            return ResponseEnvelope(
                response_type=outcome.kind,
                message=outcome.kind,
                handled_by="recipe_handler",
            )

        for turn in range(3):
            await application.handle(
                TurnRequest(
                    utterance=_PARTY_REQUEST,
                    thread_id=thread_id,
                    channel="web",
                    trace_id=f"repeat-party-{turn}",
                ),
                runtime_loader=runtime_loader,
                handlers={"recipe_handler": recipe_handler},
            )
            persisted = await service.load_task_state_record(thread_id)
            persisted_pending.append(persisted.state.pending_search_clarification)

        assert observed == [
            (None, True, "menu_plan", None, [], []),
            (None, True, "menu_plan", None, [], []),
            (None, True, "menu_plan", None, [], []),
        ]
        assert persisted_pending == [None, None, None]
        assert len(menu_plan_requests) == 3
        assert all(request.party_size == 11 for request in menu_plan_requests)
        assert all(request.constraints_confirmed for request in menu_plan_requests)
        assert all(not request.cuisines for request in menu_plan_requests)
        assert all(not request.flavors for request in menu_plan_requests)
        final_state = (await service.load_task_state_record(thread_id)).state
        final_data = final_state.to_dict()
        empty_data = ConversationTaskState().to_dict()
        final_data.pop("updated_at", None)
        empty_data.pop("updated_at", None)
        assert final_data == empty_data

    asyncio.run(run())


def test_direct_recommendation_cannot_skip_unknown_party_constraints(monkeypatch):
    """“直接推荐”只能跳过软偏好；多人安全信息未知时仍须先确认一次。"""

    party_request = "今晚有10个朋友来我家吃饭，直接推荐一个食谱清单"
    no_constraints_reply = "所有人都没什么忌口"
    menu_plan_requests: list[router.SearchRequest] = []

    async def classify(question: str, routing_context=None) -> IntentResult:
        if question == party_request:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "k": ["朋友聚餐"],
                "s": {
                    "q": "朋友聚餐",
                    "dish": [],
                    "ingredient": [],
                    "cuisine": [],
                    "flavor": [],
                    "method": [],
                    "scene": ["朋友聚餐"],
                    "meal": ["晚餐"],
                    "diet": [],
                    "exclude": [],
                    "avoid": [],
                },
                "signals": {"finalize": True},
            }
        else:
            assert question == no_constraints_reply
            raw = {
                "r": True,
                "c": "recipe_search",
                "k": [],
                "s": {
                    "q": "",
                    "dish": [],
                    "ingredient": [],
                    "cuisine": [],
                    "flavor": [],
                    "method": [],
                    "scene": [],
                    "meal": [],
                    "diet": [],
                    "exclude": [],
                    "avoid": [],
                },
                "signals": {
                    "no_constraints": True,
                    "constraint_reply": True,
                },
            }
        return IntentResult.from_raw(
            raw,
            original_text=question,
            routing_context=routing_context,
        )

    async def menu_plan(request, *, lang: str, on_search_start=None) -> dict:
        del lang, on_search_start
        menu_plan_requests.append(request)
        return {
            "success": True,
            "results": [
                {
                    "id": "safe-party-1",
                    "score": 0.9,
                    "metadata": {
                        "recipe_id": "safe-party-1",
                        "name": "已核验聚餐菜",
                        "ingredients": ["鸡肉"],
                        "tags": ["家常"],
                    },
                }
            ],
            "_menu_plan": {"queries": [{"query": "朋友聚餐"}]},
        }

    async def narrative(**_kwargs) -> RecommendationNarrative:
        return RecommendationNarrative("", "", {}, "")

    async def clarification(*_args, dimension=None, **_kwargs) -> str:
        return f"clarify:{dimension}"

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)
    monkeypatch.setattr(router, "generate_recommendation_narrative", narrative)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarification)

    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        application = TurnApplicationService(task_state_repository=repository)
        thread_id = "web:direct-party-safety-boundary"
        observed: list[tuple[str | None, bool, str, str | None]] = []
        persisted_dimensions: list[str | None] = []

        async def runtime_loader(_context):
            raise AssertionError("最小路由回归不应加载额外运行时")

        async def recipe_handler(context) -> ResponseEnvelope:
            pending = dialogue_state.get_search_clarification(thread_id)
            outcome = await router.route_fast_path(
                context.request.utterance,
                pending_search_request=(pending or {}).get("request"),
                pending_clarification_dimension=(pending or {}).get("dimension"),
                pending_asked_dimensions=list(
                    (pending or {}).get("asked_dimensions") or []
                ),
                routing_context=RoutingContext(
                    pending_clarification=pending,
                    current_task="recipe_search" if pending else None,
                    channel="web",
                ),
            )
            assert outcome.search_request is not None
            observed.append(
                (
                    (pending or {}).get("dimension"),
                    outcome.search_request.constraints_confirmed,
                    outcome.kind,
                    outcome.clarification_dimension,
                )
            )
            if outcome.kind == "clarify":
                dimension = str(outcome.clarification_dimension or "")
                dialogue_state.set_search_clarification(
                    thread_id,
                    outcome.search_request.model_dump(),
                    dimension=dimension,
                    asked_dimensions=[dimension],
                    lang=outcome.lang,
                )
            else:
                dialogue_state.clear_search_clarification(thread_id)
            return ResponseEnvelope(
                response_type=outcome.kind,
                message=outcome.kind,
                handled_by="recipe_handler",
            )

        for turn, utterance in enumerate((party_request, no_constraints_reply)):
            await application.handle(
                TurnRequest(
                    utterance=utterance,
                    thread_id=thread_id,
                    channel="web",
                    trace_id=f"direct-party-safety-{turn}",
                ),
                runtime_loader=runtime_loader,
                handlers={"recipe_handler": recipe_handler},
            )
            persisted = await service.load_task_state_record(thread_id)
            persisted_dimensions.append(
                (persisted.state.pending_search_clarification or {}).get("dimension")
            )

        assert observed == [
            (None, False, "clarify", "party_constraints"),
            ("party_constraints", True, "menu_plan", None),
        ]
        assert persisted_dimensions == ["party_constraints", None]
        assert len(menu_plan_requests) == 1
        assert menu_plan_requests[0].constraints_confirmed is True
        assert menu_plan_requests[0].party_size == 11
        assert not menu_plan_requests[0].flavors

    asyncio.run(run())


def test_full_no_constraints_statement_can_be_confirmed_without_pending():
    request = router.SearchRequest()
    intent = IntentResult(is_no_constraints=True)

    standalone = router._sanitize_no_constraints_followup(
        request,
        "所有人都没什么忌口",
        None,
        intent=intent,
    )

    assert standalone.constraints_confirmed is True


def test_bare_no_still_requires_party_constraints_pending_dimension():
    request = router.SearchRequest.from_intent_raw({}, original_text="没有")
    intent = IntentResult(is_no_constraints=True)

    standalone = router._sanitize_no_constraints_followup(
        request,
        "没有",
        None,
        intent=intent,
    )
    contextual = router._sanitize_no_constraints_followup(
        request,
        "没有",
        "party_constraints",
        intent=intent,
    )

    assert standalone.constraints_confirmed is False
    assert contextual.constraints_confirmed is True
