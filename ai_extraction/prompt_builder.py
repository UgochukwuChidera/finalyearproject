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
