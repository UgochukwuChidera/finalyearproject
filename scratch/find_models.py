import json

with open('openrouter_models.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

models = data.get('data', [])

print("Nemotron models found:")
for m in models:
    model_id = m.get('id', '').lower()
    if 'nemotron' in model_id:
        print(f"ID: {m['id']} | Name: {m['name']}")
