"""
Test extraction flow - enter file path and config when prompted.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from project.orchestrator import DAPEOrchestrator
from project.template_registry import TemplateRegistry


def list_json_files(directory):
    """List all .json config files in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = list(directory.glob("*.json"))
    return sorted(files, key=lambda x: x.name)


def choose_config():
    """Interactive config file chooser."""
    config_dir = "configs"
    print(f"\n=== Select Config File ===")
    print(f"Config directory: {config_dir}")
    print("-" * 40)
    
    files = list_json_files(config_dir)
    if not files:
        print(f"No .json files found in {config_dir}")
        return None
    
    print("Available configs:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    print("  0. Enter custom path manually")
    
    while True:
        choice = input("\nSelect config number: ").strip()
        if not choice:
            continue
        try:
            idx = int(choice)
            if idx == 0:
                path = input("Enter full config path: ").strip()
                if Path(path).exists() and path.endswith(".json"):
                    return path
            elif 1 <= idx <= len(files):
                return str(files[idx - 1])
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def choose_image():
    """Interactive image file chooser."""
    image_dir = "form"
    print(f"\n=== Select Image File ===")
    print(f"Image directory: {image_dir}")
    print("-" * 40)
    
    directory = Path(image_dir)
    if not directory.exists():
        print(f"Directory {image_dir} not found.")
        return input("Enter image path manually: ").strip()
    
    files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff")) + list(directory.glob("*.png"))
    files = sorted(files, key=lambda x: x.name)
    
    if not files:
        print(f"No image files found in {image_dir}")
        return input("Enter image path manually: ").strip()
    
    print("Available images (showing first 20):")
    for i, f in enumerate(files[:20], 1):
        print(f"  {i}. {f.name}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
    print("  0. Enter custom path manually")
    
    while True:
        choice = input("\nSelect image number: ").strip()
        if not choice:
            continue
        try:
            idx = int(choice)
            if idx == 0:
                path = input("Enter full image path: ").strip()
                if Path(path).exists():
                    return path
                print("File not found.")
            elif 1 <= idx <= len(files):
                return str(files[idx - 1])
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def sanitize(obj):
    """Convert non-JSON serializable objects."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj


def main():
    print("=== Extraction Flow Test ===")
    print("This tool extracts fields from forms using config files.")
    print("Config files define bounding boxes for each field.\n")
    
    image_path = choose_image()
    config_path = choose_config()
    
    if not image_path or not config_path:
        print("Error: Image or config not selected.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    fields = config_data.get("fields", [])
    template_path = config_data.get("template_path", "templates/registry.json")
    
    print(f"\n=== Processing ===")
    print(f"Image: {Path(image_path).name}")
    print(f"Config: {Path(config_path).name}")
    print(f"Fields defined in config: {len(fields)}")
    print("-" * 50)
    
    print("\nConfig fields overview:")
    for field in fields[:10]:
        bbox = field.get("bounding_box", {})
        print(f"  - {field.get('name')}: x={bbox.get('x')}, y={bbox.get('y')}, w={bbox.get('w')}, h={bbox.get('h')}")
    if len(fields) > 10:
        print(f"  ... and {len(fields) - 10} more fields")
    print("-" * 50)

    print("\nRunning extraction... (this may take a moment)")
    
    template_id = "student_academic_record"
    
    orch = DAPEOrchestrator(
        registry_path="templates/registry.json",
        output_dir="outputs",
        enable_hitl=False
    )
    result = orch.process(image_path, template_id)

    validated = result.get("structured_output", {}).get("fields", {})
    extracted = result.get("structured_output", {})

    print("\n=== Extracted Fields (JSON) ===")
    validated_clean = sanitize(validated)
    print(json.dumps(validated_clean, indent=2, ensure_ascii=False))

    fields_data = extracted.get("fields", [])
    null_count = sum(1 for f in fields_data if f.get("value") in ("", None, False) or f.get("value") is None)
    print(f"\nNull/Empty value count: {null_count}/{len(fields_data)}")

    save = input("\nSave raw JSON to file? (y/n): ").strip().lower()
    if save == "y":
        output_path = input("Output file path: ").strip()
        output = {
            "image_path": image_path,
            "config_path": config_path,
            "validated_fields": validated_clean,
            "extracted_fields": sanitize(fields_data),
            "stats": sanitize(result.get("stats", {})),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()