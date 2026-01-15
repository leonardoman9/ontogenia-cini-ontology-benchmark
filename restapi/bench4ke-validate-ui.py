from flask import Flask, render_template_string, request, redirect, url_for, flash
import requests, os, json
from dotenv import load_dotenv
load_dotenv()
API_TIMEOUT = int(os.getenv("CQ_API_TIMEOUT", "3600"))
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

API_BASE = os.environ.get("CQ_API_URL", "http://127.0.0.1:8000")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "changeme")

TEMPLATE = """
<!doctype html>
<title>Bench4KE</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
      <link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

<div class="container py-4">
  <h1 class="mb-4">Bench4KE Validator</h1>
  <div class="d-flex gap-3 mb-4">
  <a class="btn btn-outline-dark"
     href="https://github.com/fossr-project/ontogenia-cini/tree/main"
     target="_blank" rel="noopener">
    <i class="bi bi-github me-2"></i>GitHub
  </a>

  <a class="btn btn-outline-primary"
     href="https://docs.google.com/forms/d/e/1FAIpQLSfpYHGzA2r0wKCq0xEVIkPBKKol6umiKn1URAc17f709DKMKg/viewform?usp=dialog"
     target="_blank" rel="noopener">
    Tell us what you think
  </a>
</div>


  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-warning mt-3">{{ messages[0] }}</div>
    {% endif %}
  {% endwith %}

  <ul class="nav nav-tabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if active_tab == 'cq' %}active{% endif %}" id="cq-tab" data-bs-toggle="tab" data-bs-target="#cq-panel" type="button" role="tab">CQ Validation</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if active_tab == 'ontology' %}active{% endif %}" id="ontology-tab" data-bs-toggle="tab" data-bs-target="#ontology-panel" type="button" role="tab">Ontology Benchmark</button>
    </li>
  </ul>

  <div class="tab-content border border-top-0 p-3 bg-white">
    <div class="tab-pane fade {% if active_tab == 'cq' %}show active{% endif %}" id="cq-panel" role="tabpanel">
      <form class="mt-2" method="post" action="{{ url_for('validate') }}">
        <div class="mb-3">
          <label for="external_url" class="form-label">CQ Generator API URL:</label>
          <input class="form-control" type="url" id="external_url" name="external_url" placeholder="https://.../newapi/" required>
        </div>

        <button class="btn btn-primary" type="submit">Validate with Default Dataset</button>
      </form>

      {% if result %}
        <hr>
        <h2 class="h4">Validation Result</h2>
        <pre class="border p-3 bg-light">{{ result | tojson(indent=2) }}</pre>

        {% for r in result.validation_results %}
          {% if r.Cosine_Heatmap %}
            <h5 class="mt-4">Cosine heat-map</h5>
            <img class="img-fluid" src="{{ r.Cosine_Heatmap }}">
          {% endif %}
          {% if r.Jaccard_Heatmap %}
            <h5 class="mt-4">Jaccard heat-map</h5>
            <img class="img-fluid" src="{{ r.Jaccard_Heatmap }}">
          {% endif %}
        {% endfor %}
      {% endif %}
    </div>

    <div class="tab-pane fade {% if active_tab == 'ontology' %}show active{% endif %}" id="ontology-panel" role="tabpanel">
      <form class="mt-2" method="post" action="{{ url_for('run_ontology') }}">
        <div class="mb-3">
          <label for="ontology_url" class="form-label">Ontology Generator API URL:</label>
          <input class="form-control" type="url" id="ontology_url" name="ontology_url" placeholder="https://.../generate_ontology" required>
        </div>
        <div class="mb-3">
          <label for="system" class="form-label">System filter (optional):</label>
          <input class="form-control" type="text" id="system" name="system" placeholder="ontogenia | domain-ontogen | neon-gpt | all">
        </div>
        <div class="mb-3">
          <label for="evaluation_mode" class="form-label">Evaluation mode:</label>
          <input class="form-control" type="text" id="evaluation_mode" name="evaluation_mode" value="all">
        </div>
        <div class="mb-3">
          <label for="max_items" class="form-label">Max items (optional):</label>
          <input class="form-control" type="number" id="max_items" name="max_items" min="0" step="1" placeholder="0 = all">
        </div>
        <div class="mb-3">
          <label for="llm_eval_model" class="form-label">LLM eval model (optional):</label>
          <input class="form-control" type="text" id="llm_eval_model" name="llm_eval_model" placeholder="gpt-4o">
        </div>

        <button class="btn btn-primary" type="submit">Run Ontology Benchmark</button>
      </form>

      {% if ontology_result %}
        <hr>
        <h2 class="h4">Ontology Benchmark Result</h2>
        <pre class="border p-3 bg-light">{{ ontology_result | tojson(indent=2) }}</pre>
      {% endif %}
    </div>
  </div>
</div>
"""

# ---- routes ----
@app.route("/")
def index():
    return render_template_string(TEMPLATE, result=None, ontology_result=None, active_tab="cq")

@app.post("/validate")
def validate():
    external_url = request.form.get("external_url", "").strip()
    if not external_url:
        flash("Please provide the CQ generator API URL.")
        return redirect(url_for("index"))

    data = {
        "use_default_dataset": "true",
        "external_service_url": external_url,
        "validation_mode": "all",
        "model": DEFAULT_OPENAI_MODEL,
        "save_results": "true"
    }
    api_key = request.form.get("api_key", os.getenv("EXTERNAL_CQ_API_KEY", "")).strip()
    if api_key:
        data["api_key"] = api_key

    try:
        connect_tm = 10
        read_tm = None if API_TIMEOUT == 0 else API_TIMEOUT
        resp = requests.post(
            f"{API_BASE}/validate/",
            data=data,
            timeout=(connect_tm, read_tm)
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        flash(f"Error contacting Validator API: {exc}")
        return redirect(url_for("index"))

    return render_template_string(TEMPLATE, result=result, ontology_result=None, active_tab="cq")

@app.post("/ontology")
def run_ontology():
    external_url = request.form.get("ontology_url", "").strip()
    if not external_url:
        flash("Please provide the ontology generator API URL.")
        return redirect(url_for("index"))

    payload = {
        "use_default_dataset": True,
        "evaluation_mode": (request.form.get("evaluation_mode") or "all").strip(),
        "external_service_url": external_url,
        "save_results": True,
    }
    system = (request.form.get("system") or "").strip()
    if system:
        payload["system"] = system
    max_items = (request.form.get("max_items") or "").strip()
    if max_items:
        try:
            payload["max_items"] = int(max_items)
        except ValueError:
            flash("Max items must be an integer.")
            return redirect(url_for("index"))
    llm_eval_model = (request.form.get("llm_eval_model") or "").strip()
    if llm_eval_model:
        payload["llm_eval_model"] = llm_eval_model

    try:
        connect_tm = 10
        read_tm = None if API_TIMEOUT == 0 else API_TIMEOUT
        resp = requests.post(
            f"{API_BASE}/ontology/run",
            json=payload,
            timeout=(connect_tm, read_tm)
        )
        resp.raise_for_status()
        ontology_result = resp.json()
    except Exception as exc:
        flash(f"Error contacting Ontology API: {exc}")
        return redirect(url_for("index"))

    return render_template_string(
        TEMPLATE,
        result=None,
        ontology_result=ontology_result,
        active_tab="ontology",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
