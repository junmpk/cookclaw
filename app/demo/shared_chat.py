"""Public data adapter; routing, memory and selection remain in the shared runtime."""
_store = None
# Display translations of the eight imported source titles, not new recipes.
TITLES_ZH = {
    "chicken-and-broccoli-bake": "鸡肉西兰花烤饭",
    "broccoli-omelet": "西兰花煎蛋",
    "bean-and-vegetable-salad": "豆类蔬菜沙拉",
    "broccoli-baked-potatoes": "西兰花烤土豆",
    "apple-banana-salad-peanuts": "苹果香蕉花生沙拉",
    "baked-eggs-cheese": "芝士烤蛋",
    "quick-chicken-vegetable-soup": "快手鸡肉蔬菜汤",
    "vegetable-soup-chicken": "鸡肉蔬菜汤",
}


def bind_recipe_store(store):
    global _store
    _store = store


async def search_public_recipes(query, top_k=3, lang=None):
    if _store is None:
        raise RuntimeError("Public recipe store is not bound")
    recipes = await _store.search(query, 16)
    rows = []
    for recipe in recipes:
        name = TITLES_ZH.get(recipe.id, recipe.name) if lang != "en" else recipe.name
        detail = {
            "schema_version": "recipe_detail_v1", "recipe_id": recipe.id,
            "name": name, "language": "en", "source": recipe.source,
            "ingredients": [{"name": x, "group": "main"} for x in recipe.ingredients],
            "steps": [{"number": i, "type": "manual", "description": x,
                       "parameters": []} for i, x in enumerate(recipe.steps, 1)],
            "nutrition": recipe.nutrition,
        }
        rows.append({"id": recipe.id, "metadata": {
            "recipe_id": recipe.id, "name": name,
            "image_url": recipe.image_url or "",
            "ingredients": recipe.ingredients, "tags": [recipe.kind],
            "facets": {"meal": [recipe.kind]},
            "description": "原名：" + recipe.name + "；公开来源：" + recipe.source,
            "recipe_detail": detail, "device_code": None,
        }})
    rows = rows[:max(1, min(top_k, 16))]
    return {"success": True, "query": query, "count": len(rows), "results": rows}
