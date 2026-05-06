import json

with open('openrouter_models.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for m in data.get('data', []):
    name = m.get('name', '').lower()
    mid = m.get('id', '').lower()
    if 'qwen' in mid:
        print(f"ID: {m['id']} | Name: {m['name']} | Modality: {m.get('architecture', {}).get('modality')}")
