from typing import Dict


def build_multi_image_prompt(config: dict, full_image_bytes: bytes, critical_crops: Dict[str, bytes]) -> list[dict]:
    fields = config.get("fields", [])
    critical_names = [f.get("name") for f in fields if f.get("critical")]
    non_critical_names = [f.get("name") for f in fields if not f.get("critical")]

    instructions = []
    instructions.append(
        {
            "prompt": (
                "You are extracting data from a filled form. "
                "Return ONLY JSON with shape {\"fields\": {\"field_name\": value}, \"meta\": {\"C_lp\": {\"field_name\": number}}}. "
                "For checkboxes, return true/false. "
                "For non-critical static fields, extract what appears in the full form."
            ),
            "image": full_image_bytes,
            "field_name": "__full_form__",
        }
    )

    if non_critical_names:
        instructions.append(
            {
                "prompt": "Non-critical fields to extract from full image: " + ", ".join(non_critical_names),
                "image": full_image_bytes,
                "field_name": "__non_critical__",
            }
        )

    for field in fields:
        if not field.get("critical"):
            continue
        name = field.get("name")
        label = field.get("label_hint", name)
        if name not in critical_crops:
            continue
        instructions.append(
            {
                "prompt": (
                    f"Image above is an ink-only crop for critical handwritten field '{name}' ({label}). "
                    "Transcribe exactly including spelling and spacing. "
                    f"Set fields.{name} to the transcription."
                ),
                "image": critical_crops[name],
                "field_name": name,
            }
        )

    return instructions


def build_discovery_prompt() -> str:
    return (
        "Analyze this blank form template. Identify every entry field, checkbox, and section. "
        "For each field, return: "
        "1. A unique 'name' (snake_case). "
        "2. A human-readable 'label'. "
        "3. A 'type' ('text' or 'checkbox'). "
        "4. 'critical' (boolean, set to true if it contains sensitive ID info or medical data). "
        "5. 'bounding_box' as {x, y, w, h} where coordinates are NORMALIZED (0 to 1000). "
        "For example, a box in the exact center would be {x: 500, y: 500, w: 100, h: 100}. "
        "Return the result as a JSON object with a 'fields' list."
    )
