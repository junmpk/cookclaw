"""Conservative display cleanup for legacy English ingredient token lists.

Some imported English records were split on every space before entering Milvus.
This module only rejoins common adjacent ingredient phrases and removes preparation
fragments for display. It does not add ingredients that are absent from the source.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


_PHRASES = {
    ("beef", "short", "ribs"): "beef short ribs",
    ("green", "bell", "pepper"): "green bell pepper",
    ("red", "bell", "pepper"): "red bell pepper",
    ("red", "kidney", "bean"): "red kidney beans",
    ("beef", "minced"): "minced beef",
    ("slices", "beef"): "beef slices",
    ("beef", "slices"): "beef slices",
    ("beef", "brisket"): "beef brisket",
    ("beef", "tenderloin"): "beef tenderloin",
    ("beef", "steak"): "beef steak",
    ("beef", "cheek"): "beef cheek",
    ("ground", "beef"): "ground beef",
    ("corn", "kernels"): "corn kernels",
    ("kidney", "bean"): "kidney beans",
    ("mustard", "paste"): "mustard",
    ("tomato", "paste"): "tomato paste",
    ("tomato", "sauce"): "tomato sauce",
    ("chipotle", "paste"): "chipotle paste",
    ("soy", "sauce"): "soy sauce",
    ("light", "soy", "sauce"): "light soy sauce",
    ("dark", "soy", "sauce"): "dark soy sauce",
    ("hot", "sauce"): "hot sauce",
    ("vegetable", "oil"): "vegetable oil",
    ("olive", "oil"): "olive oil",
    ("sesame", "oil"): "sesame oil",
    ("cooking", "wine"): "cooking wine",
    ("white", "wine"): "white wine",
    ("red", "wine"): "red wine",
    ("black", "pepper"): "black pepper",
    ("white", "pepper"): "white pepper",
    ("cumin", "powder"): "cumin powder",
    ("garlic", "powder"): "garlic powder",
    ("garlic", "cloves"): "garlic",
    ("star", "anise"): "star anise",
    ("bay", "leaf"): "bay leaf",
    ("bay", "leaves"): "bay leaves",
    ("spring", "onions"): "spring onions",
    ("egg", "white"): "egg white",
    ("mashed", "potato"): "mashed potato",
    ("corn", "tortillas"): "corn tortillas",
    ("chicken", "wings"): "chicken wings",
    ("brown", "sugar"): "brown sugar",
    ("rock", "sugar"): "rock sugar",
}

_NOISE = {
    "a", "an", "and", "or", "of", "to", "for", "with", "into", "in", "the", "only",
    "approximately", "approx", "about", "plus", "extra",
    "cut", "chopped", "diced", "sliced", "peeled", "trimmed", "washed",
    "soaked", "unsoaked", "roughly", "finely", "fine", "small", "large",
    "quartered", "cored", "softened", "boneless",
    "chunks", "chunk", "pieces", "piece", "slices", "slice", "lengths",
    "tsp", "tbsp", "teaspoon", "tablespoon", "pc", "pcs", "cm",
    "taste", "pinch", "handful", "ground", "cooking", "drinking",
    "paste", "powder", "cloves", "flesh", "fresh", "raw",
    "red", "green", "white", "brown",
    "-", "(", ")",
}

_PRIMARY_MARKERS = (
    "beef", "chicken", "pork", "lamb", "mutton", "fish",
    "shrimp", "prawn", "tofu", "egg",
)


def clean_english_ingredient_tokens(
    values: Iterable[object],
    *,
    limit: int | None = None,
) -> list[str]:
    """Return readable source-grounded ingredients from tokenized English values."""
    raw = [str(value or "").strip() for value in values]
    raw = [value for value in raw if value and re.search(r"[A-Za-z]", value)]
    if not raw:
        return []

    # Proper phrases from clean API data should remain untouched. The legacy defect
    # is recognizable because nearly every element is a single token.
    single_token_ratio = sum(" " not in value for value in raw) / len(raw)
    if single_token_ratio < 0.8:
        result = list(dict.fromkeys(raw))
        return result[:limit] if limit else result

    lowered = [re.sub(r"[^a-z]+", "", value.lower()) for value in raw]
    out: list[str] = []
    index = 0
    while index < len(raw):
        matched = False
        for width in (3, 2):
            key = tuple(lowered[index:index + width])
            phrase = _PHRASES.get(key)
            if phrase:
                if phrase not in out:
                    out.append(phrase)
                index += width
                matched = True
                break
        if matched:
            continue

        key = lowered[index]
        value = raw[index].strip(" ,;:()[]")
        if (
            key
            and key not in _NOISE
            and len(key) > 1
            and value
            and value.lower() not in {item.lower() for item in out}
        ):
            out.append(value)
        index += 1
    primary = [
        item for item in out
        if any(re.search(rf"\b{marker}\b", item.lower()) for marker in _PRIMARY_MARKERS)
    ]
    secondary = [item for item in out if item not in primary]
    result = [*primary, *secondary]
    return result[:limit] if limit else result
