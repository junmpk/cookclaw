"""主阅读入口：并行研究 → 菜单 → 校验/定向修订 → 饮食审阅 → 人工确认。"""
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .details import hydrate_schedule_details
from .models import Brief, ComplexityProfile, DemoState, Recipe
from .retrieval import vector
from .validation import validate_menu, validate_plan


def _apply_slot_replacement(menu: dict, candidates: list[Recipe], replacement: dict | None) -> dict:
    """保留原菜单其它槽位，只替换用户指定的一格。"""
    if not replacement:
        return menu
    try:
        index = int(replacement["slot_index"])
    except (KeyError, TypeError, ValueError):
        return menu
    previous_ids = [str(item) for item in replacement.get("previous_ids") or []]
    if index < 0 or index >= len(previous_ids):
        return menu
    old_id = previous_ids[index]
    catalog = {recipe.id: recipe for recipe in candidates}
    old = catalog.get(old_id)
    if old is None:
        return menu
    used = set(previous_ids)
    proposed = {str(item) for item in menu.get("recipe_ids") or []}
    alternatives = [
        recipe for recipe in candidates
        if recipe.kind == old.kind and recipe.id not in used and recipe.id not in proposed
        and not any(
            term.casefold() in " ".join(recipe.ingredients).casefold()
            for term in (replacement.get("exclusions") or [])
        )
    ]
    if not alternatives:
        return {**menu, "recipe_ids": previous_ids}
    request = str(replacement.get("request") or "").strip()
    if request:
        query_vector = vector(request)

        def relevance(recipe: Recipe) -> float:
            candidate_vector = vector(
                " ".join([recipe.name, *recipe.ingredients])
            )
            return sum(
                left * right
                for left, right in zip(query_vector, candidate_vector)
            )

        alternatives.sort(key=relevance, reverse=True)
    selected = list(previous_ids)
    selected[index] = alternatives[0].id
    return {
        **menu,
        "recipe_ids": selected,
        "explanation": (
            "已按用户指定槽位和调整要求替换，"
            "其余菜品和汤保持不变。"
        ),
    }


def build_graph(
    agents,
    saver,
    storage,
    run_id: str,
    emit,
    complexity_profile: ComplexityProfile | None = None,
):
    selected_agents = set(
        (complexity_profile or ComplexityProfile(
            level="coordinated",
            selected_agents=["research", "dietary", "menu"],
        )).selected_agents
    )

    def recipes(state):
        return [Recipe.model_validate(item) for item in state.get("candidates", [])]

    def observed(role, name, operation):
        async def node(state):
            started = time.monotonic()
            await emit(role, "node_start", {"node": name})
            result = await operation(state)
            await emit(role, "node_end", {"node": name, "duration_ms": round((time.monotonic() - started) * 1000), "output": result})
            return result
        return node

    async def research(state):
        attempt = state.get("research_count", 0) + 1
        brief = Brief(**state["brief"])
        preloaded = [
            Recipe.model_validate(item)
            for item in state.get("preloaded_candidates", [])
        ]
        revision_request = str(
            (state.get("replacement") or {}).get("request") or ""
        ).strip()
        if revision_request:
            combined_request = (
                f"{brief.request}；局部调整：{revision_request}"
            )[:2000]
            brief = brief.model_copy(
                update={"request": combined_request}
            )
        result = await agents.research(
            brief,
            state.get("issues", []),
            attempt,
            preloaded_candidates=(preloaded if attempt == 1 else None),
        )
        return {"candidates": result, "research_count": attempt}

    async def dietary(state):
        return {"dietary": await agents.diet(Brief(**state["brief"]))}

    async def inventory(state):
        result = await agents.inventory(Brief(**state["brief"]), recipes(state))
        return {"inventory": result}

    async def menu(state):
        result = await agents.plan(
            Brief(**state["brief"]),
            recipes(state),
            state["dietary"],
            state.get("issues", []),
            state.get("inventory"),
        )
        result = _apply_slot_replacement(result, recipes(state), state.get("replacement"))
        return {"menu": result, "revision_count": state.get("revision_count", 0) + 1}

    async def schedule(state):
        result = await agents.schedule(
            Brief(**state["brief"]),
            recipes(state),
            state["menu"],
        )
        return {"schedule": result}

    async def hydrate_details(state):
        hydrated, summary = hydrate_schedule_details(
            Brief(**state["brief"]),
            recipes(state),
            [str(item) for item in state["menu"]["recipe_ids"]],
        )
        return {
            "candidates": [recipe.model_dump() for recipe in hydrated],
            "detail_hydration": summary,
        }

    async def validate(state):
        issues = validate_menu(Brief(**state["brief"]), recipes(state), state["menu"]["recipe_ids"])
        replacement = state.get("replacement") or {}
        previous_ids = [str(item) for item in replacement.get("previous_ids") or []]
        index = replacement.get("slot_index")
        if (
            previous_ids
            and isinstance(index, int)
            and index < len(previous_ids)
            and state["menu"]["recipe_ids"][index : index + 1]
            == previous_ids[index : index + 1]
        ):
            issues.append(f"第 {index + 1} 道暂时没有可核验的替代食谱")
        return {"issues": issues}

    def validation_route(state):
        if not state["issues"]:
            return "review"
        if state["revision_count"] >= 3:
            return "blocked"
        if any("需要" in issue for issue in state["issues"]) and state["research_count"] < 2:
            return "refetch"
        return "menu"

    async def review(state):
        result = await agents.review(
            Brief(**state["brief"]),
            recipes(state),
            state["menu"],
            state.get("inventory"),
            state.get("schedule"),
        )
        return {"review": result, "issues": result["findings"] if result["needs_revision"] else []}

    def review_route(state):
        if not state["review"]["needs_revision"]:
            return "final_validate"
        return "menu" if state["revision_count"] < 3 else "blocked"

    async def final_validate(state):
        report = validate_plan(
            Brief(**state["brief"]),
            recipes(state),
            state["menu"],
            dietary=state.get("dietary"),
            inventory=state.get("inventory"),
            schedule=state.get("schedule"),
            review=state.get("review"),
        )
        return {
            "plan_validation": report.model_dump(),
            "issues": list(report.blocking_issues),
        }

    def final_validation_route(state):
        report = state.get("plan_validation") or {}
        return "blocked" if report.get("blocking_issues") else "approval"

    async def blocked(state):
        return {"status": "blocked", "approved": False}

    def approval(state):
        # This node restarts on resume. No side effects before interrupt().
        answer = interrupt({"version": state["version"], "recipe_ids": state["menu"]["recipe_ids"],
                            "message": "请核对方案及未知项，再确认 Mock 执行。"})
        if not isinstance(answer, dict) or answer.get("version") != state["version"]:
            raise ValueError("确认版本失效")
        accepted = answer.get("approved") is True
        return {"approved": accepted, "status": "approved" if accepted else "cancelled"}

    async def execute(state):
        # Revalidate on resume; a checkpoint is never an authorization bypass.
        report = validate_plan(
            Brief(**state["brief"]),
            recipes(state),
            state["menu"],
            dietary=state.get("dietary"),
            inventory=state.get("inventory"),
            schedule=state.get("schedule"),
            review=state.get("review"),
        )
        if report.blocking_issues or not state.get("approved"):
            raise ValueError("方案未通过执行前校验")
        result = storage.execute_mock(f"{run_id}:v{state['version']}", state["menu"]["recipe_ids"])
        return {"execution": result, "status": "completed"}

    builder = StateGraph(DemoState)
    builder.add_node("research", observed("research", "research", research))
    builder.add_node("dietary", observed("diet", "dietary", dietary))
    builder.add_node("menu", observed("menu", "menu", menu))
    builder.add_node("validate", observed("system", "validate", validate))
    builder.add_node("refetch", observed("research", "refetch", research))
    builder.add_node("review", observed("diet", "review", review))
    builder.add_node(
        "final_validate",
        observed("system", "final_validate", final_validate),
    )
    builder.add_node("approval", approval)
    builder.add_node("execute", observed("system", "execute", execute))
    builder.add_node("blocked", observed("system", "blocked", blocked))
    builder.add_edge(START, "research")
    builder.add_edge(START, "dietary")
    if "inventory" in selected_agents:
        builder.add_node("inventory", observed("inventory", "inventory", inventory))
        builder.add_node(
            "inventory_refresh",
            observed("inventory", "inventory_refresh", inventory),
        )
        builder.add_edge("research", "inventory")
        builder.add_edge(["inventory", "dietary"], "menu")
    else:
        builder.add_edge(["research", "dietary"], "menu")
    if "scheduler" in selected_agents:
        builder.add_node(
            "hydrate_details",
            observed("system", "hydrate_details", hydrate_details),
        )
        builder.add_node("schedule", observed("scheduler", "schedule", schedule))
        builder.add_edge("menu", "hydrate_details")
        builder.add_edge("hydrate_details", "schedule")
        builder.add_edge("schedule", "validate")
    else:
        builder.add_edge("menu", "validate")
    builder.add_conditional_edges("validate", validation_route)
    builder.add_edge(
        "refetch",
        "inventory_refresh" if "inventory" in selected_agents else "menu",
    )
    if "inventory" in selected_agents:
        builder.add_edge("inventory_refresh", "menu")
    builder.add_conditional_edges("review", review_route)
    builder.add_conditional_edges("final_validate", final_validation_route)
    builder.add_conditional_edges("approval", lambda state: "execute" if state["approved"] else END)
    builder.add_edge("execute", END)
    builder.add_edge("blocked", END)
    return builder.compile(checkpointer=saver)
