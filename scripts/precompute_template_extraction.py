import argparse
import json
from pathlib import Path

import cv2

from ai_extraction import GeminiClient


def _to_png_bytes(image):
    ok, buff = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('Failed to encode template image')
    return buff.tobytes()


def main():
    parser = argparse.ArgumentParser(description='Precompute blank template extraction JSON')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open('r', encoding='utf-8') as fh:
        config = json.load(fh)

    template_path = Path(config['template_path'])
    image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Cannot load template: {template_path}')

    fields = config.get('fields', [])
    names = [f.get('name') for f in fields]
    prompt = (
        'This is a blank form template. Return ONLY JSON: '
        '{"fields": {"name": "placeholder"}}. '
        'For each field listed, return placeholder text exactly as shown on template (or underscores) and do not invent user input. '
        f'Fields: {", ".join([n for n in names if n])}'
    )

    payload = GeminiClient().extract_from_images(images=[_to_png_bytes(image)], prompts=[prompt])
    out = Path(config.get('template_extraction') or (config_path.parent / f"{config.get('form_type','template')}_extraction.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f'Saved template extraction: {out}')


if __name__ == '__main__':
    main()
