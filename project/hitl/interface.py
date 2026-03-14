"""
HITL Web Review Interface
Flask server that presents flagged fields for human correction.
Opens automatically when low-confidence fields are detected.
Access at http://127.0.0.1:5050 (or configured port).
"""

import json
import threading
from flask import Flask, jsonify, render_template_string, request

_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DAPE – Field Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f4f5f7;color:#1a1a2e}
header{background:#1a1a2e;color:#fff;padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem}
header h1{font-size:1.1rem;font-weight:600}
.badge{background:#e74c3c;color:#fff;border-radius:999px;padding:.15rem .55rem;font-size:.8rem}
main{max-width:860px;margin:2rem auto;padding:0 1rem}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);padding:1.25rem 1.5rem;margin-bottom:1.2rem}
.card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.75rem}
.field-id{font-weight:700;font-size:.95rem}
.field-type{font-size:.78rem;color:#666;text-transform:uppercase;letter-spacing:.05em}
.status-badge{font-size:.75rem;padding:.2rem .5rem;border-radius:4px}
.low_confidence{background:#fff3cd;color:#856404}
.semantic_failure{background:#f8d7da;color:#842029}
.rejected{background:#f8d7da;color:#842029}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
label{font-size:.82rem;color:#555;display:block;margin-bottom:.2rem}
.current-value{background:#f4f5f7;border:1px solid #ddd;border-radius:4px;padding:.45rem .6rem;font-size:.9rem;color:#444;word-break:break-all}
input[type=text]{width:100%;border:1.5px solid #bbb;border-radius:4px;padding:.45rem .6rem;font-size:.9rem;transition:border-color .15s}
input[type=text]:focus{outline:none;border-color:#3b82f6}
.reason{font-size:.78rem;color:#888;margin-top:.4rem}
.conf{font-size:.78rem;color:#666}
footer-bar{display:block;position:sticky;bottom:0;background:#fff;border-top:1px solid #ddd;padding:1rem 1.5rem;display:flex;justify-content:flex-end;gap:.75rem}
button{cursor:pointer;border:none;border-radius:6px;padding:.55rem 1.2rem;font-size:.9rem;font-weight:600}
.btn-primary{background:#3b82f6;color:#fff}
.btn-primary:hover{background:#2563eb}
.no-flags{text-align:center;padding:3rem 1rem;color:#666}
.checkbox-wrap{display:flex;align-items:center;gap:.5rem;margin-top:.5rem}
.checkbox-wrap input[type=checkbox]{width:auto}
</style>
</head>
<body>
<header>
  <h1>DAPE — Human Review Interface</h1>
  <span class="badge">{{ fields|length }} field(s) flagged</span>
</header>
<main>
{% if fields %}
<form id="review-form">
{% for f in fields %}
<div class="card">
  <div class="card-header">
    <div><div class="field-id">{{ f.field_id }}</div>
    <div class="field-type">{{ f.field_type }}</div></div>
    <span class="status-badge {{ f.validation_status }}">{{ f.validation_status }}</span>
  </div>
  <div class="grid-2">
    <div>
      <label>Extracted value</label>
      <div class="current-value">{{ f.value if f.value != '' and f.value is not none else '(empty)' }}</div>
      <div class="reason">Reason: {{ f.validation_reason }}</div>
      <div class="conf">Confidence: {{ "%.1f%%"|format(f.confidence * 100) }}</div>
    </div>
    <div>
      <label>Corrected value</label>
      {% if f.field_type == 'checkbox' %}
        <div class="checkbox-wrap">
          <input type="checkbox" id="corr_{{ f.field_id }}" name="{{ f.field_id }}"
            {% if f.value %}checked{% endif %}>
          <label for="corr_{{ f.field_id }}" style="margin:0">Marked</label>
        </div>
      {% else %}
        <input type="text" name="{{ f.field_id }}"
               value="{{ f.value if f.value is not none else '' }}"
               placeholder="Enter corrected value…">
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
</form>
{% else %}
<div class="no-flags">✅ No fields require review.</div>
{% endif %}
</main>
<footer-bar>
  <button type="button" class="btn-primary" onclick="submitReview()">Submit corrections</button>
</footer-bar>
<script>
function submitReview(){
  const form=document.getElementById('review-form');
  if(!form){fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections:{}})});return;}
  const corrections={};
  form.querySelectorAll('input[type=text]').forEach(el=>{if(el.name)corrections[el.name]=el.value;});
  form.querySelectorAll('input[type=checkbox]').forEach(el=>{
    if(el.id&&el.id.startsWith('corr_'))corrections[el.id.replace('corr_','')]=el.checked;
  });
  fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections})})
    .then(r=>r.json()).then(()=>{
      document.body.innerHTML='<div style="text-align:center;padding:4rem;font-family:system-ui"><h2 style="color:#16a34a">✅ Corrections submitted. You may close this tab.</h2></div>';
    });
}
</script>
</body>
</html>
"""


class HITLInterface:
    def __init__(self, host="127.0.0.1", port=5050):
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._pending: list = []
        self._corrections: dict = {}
        self._done = threading.Event()
        self._started = False
        self._register_routes()

    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return render_template_string(_PAGE, fields=self._pending)

        @app.route("/fields")
        def get_fields():
            return jsonify(self._pending)

        @app.route("/submit", methods=["POST"])
        def submit():
            data = request.get_json(silent=True) or {}
            self._corrections = data.get("corrections", {})
            self._done.set()
            return jsonify({"status": "ok"})

    def run_review(self, flagged_fields: list) -> dict:
        self._pending = flagged_fields
        self._corrections = {}
        self._done.clear()
        if not self._started:
            t = threading.Thread(
                target=lambda: self._app.run(
                    host=self.host, port=self.port,
                    debug=False, use_reloader=False),
                daemon=True)
            t.start()
            self._started = True
        print(f"\n[HITL] Review interface → http://{self.host}:{self.port}")
        print(f"       {len(flagged_fields)} field(s) require attention.")
        print(f"       Waiting for submission…\n")
        self._done.wait()
        print("[HITL] Corrections received.\n")
        return dict(self._corrections)
