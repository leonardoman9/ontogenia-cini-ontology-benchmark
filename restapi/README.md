# REST API

This folder contains the Bench4KE backend (CQ validation + ontology benchmark), plus example services for CQ generation and ontology generation. It also includes a simple UI and optional tools.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create and activate the environment file:

```bash
cp .env.example .env
source .env
```

`python-dotenv` loads `.env` automatically in most services, but sourcing it keeps CLI runs consistent.

### Environment variables (summary)

Essentials:
- `OPENAI_API_KEY` (required for any real run)
- `OOPS_API_URL` only if you want OOPS enabled; leave empty to skip it

Optional with defaults:
- `OPENAI_MODEL` defaults to `gpt-4o-mini`
- `EXTERNAL_CQ_GENERATION_URL` defaults to `http://127.0.0.1:8001/newapi`
- `EXTERNAL_ONTOLOGY_SERVICE_URL` defaults to `http://127.0.0.1:8020/generate_ontology`
- `ONTOLOGY_SYSTEM` defaults to `ontogenia`

See `.env.example` for the full list (timeouts, paths, retry settings, UI config, optional providers).

## Service: Bench4KE API (FastAPI)

Start the API:

```bash
cd restapi
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Endpoint: CQ validation

`POST /validate/` (multipart/form-data)

Main fields:
- `use_default_dataset` (true/false)
- `external_service_url` (CQ generator URL)
- `validation_mode` (all|cosine|jaccard|llm)
- `model` (OpenAI model name)
- `save_results` (true/false)

Example:

```bash
curl -X POST http://127.0.0.1:8000/validate/ \
  -F use_default_dataset=true \
  -F external_service_url=http://127.0.0.1:8001/newapi \
  -F validation_mode=all \
  -F save_results=true
```

Outputs:
- Results CSV in `restapi/outputs/cq_validation/results/`
- Heatmaps in `restapi/outputs/cq_validation/heatmaps/`

### Endpoint: Ontology benchmark

`POST /ontology/run` (application/json)

Main fields:
- `use_default_dataset` (true/false)
- `dataset_path` (optional JSONL file or directory)
- `items` (optional array of items)
- `system` (ontogenia|domain-ontogen|neon-gpt|all)
- `evaluation_mode` (all|ontometrics|oops|llm or comma-separated)
- `external_service_url` (ontology adapter URL)
- `save_results` (true/false)
- `max_items` (limit items)
- `model`, `llm_eval_model` (OpenAI model names)

Example:

```bash
curl -X POST http://127.0.0.1:8000/ontology/run \
  -H "Content-Type: application/json" \
  -d '{
    "use_default_dataset": true,
    "evaluation_mode": "all",
    "external_service_url": "http://127.0.0.1:8020/generate_ontology",
    "save_results": true,
    "max_items": 5
  }'
```

Outputs (per run):
- `restapi/outputs/ontology_benchmark/runs/<run_id>/ontologies/` (TTL/OWL files)
- `restapi/outputs/ontology_benchmark/runs/<run_id>/metrics/` (ontometrics, OOPS, LLM eval)
- `restapi/outputs/ontology_benchmark/runs/<run_id>/metrics/sparql_queries/` (SPARQL)
- `restapi/outputs/ontology_benchmark/runs/<run_id>/summary.json`
- `restapi/outputs/ontology_benchmark/runs/<run_id>/run_metadata.json`

## Service: Competency Question generator

File name: `cq_generator_app.py`

Purpose: given a CSV with columns such as Scenario, Dataset, and Description, it generates a competency question per row.

Start the service:

```bash
cd restapi
python cq_generator_app.py
```

Default bind is `http://127.0.0.1:8001`

Endpoint is `POST /newapi/` with `multipart/form-data`. Expected fields are `file` as a CSV upload, `llm_provider` set to `openai` or `together` or `claude` with default `openai`, and optional sampling parameters. The response is a CSV with the gold standard and the generated question.

Example request:

```bash
curl -X POST "http://127.0.0.1:8001/newapi/" \
  -F "file=@./examples/cq_input.csv" \
  -F "llm_provider=openai" \
  --output generated_cq.csv
```

## Service: Ontology adapter

File name: `ontology_adapter.py`

Purpose: exposes `POST /generate_ontology` and adapts prompts for the required systems.

Start the service:

```bash
cd restapi
python ontology_adapter.py
```

Default bind is `http://127.0.0.1:8020`

Request body (JSON) includes:
- `system` (ontogenia|domain-ontogen|neon-gpt)
- `dataset_id` or `scenario_id`
- `competency_questions` (list)
- `user_stories` (optional)
- `constraints` (optional)
- `metadata` (optional)

Response includes `ontology.format` and `ontology.content`.

## How to add a new ontology generation system

1) **Add dataset items**
- Create a JSONL file in `datasets/ontology_generation/normalized/`.
- Each line must include `dataset_id` (or `id`) and `competency_questions` (list).
- Optionally add `system`, `scenario`, `user_stories`, `constraints`, `metadata`.

Example:
```json
{"dataset_id":"my_case_01","system":"my-system","scenario":"...","competency_questions":["Q1","Q2"]}
```

2) **Add the prompt**
- Place the prompt under `datasets/ontology_generation/raw/<my-system>/`.
- If you want to override, set `ONTOLOGY_PROMPT_FILE` in `.env`.

3) **Update the adapter**
- Add the system to `ALLOWED_SYSTEMS` in `ontology_adapter.py`.
- Add a branch in `_default_prompt_path()` for the new prompt.
- Add post-processing if the output needs cleanup.

4) **Run the adapter**
```bash
ONTOLOGY_SYSTEM=my-system python ontology_adapter.py
```

5) **Run the benchmark**
```bash
curl -X POST http://127.0.0.1:8000/ontology/run \
  -H "Content-Type: application/json" \
  -d '{
    "use_default_dataset": true,
    "system": "my-system",
    "evaluation_mode": "all",
    "external_service_url": "http://127.0.0.1:8020/generate_ontology",
    "save_results": true
  }'
```

6) **Check outputs**
- `restapi/outputs/ontology_benchmark/runs/<run_id>/ontologies/`
- `restapi/outputs/ontology_benchmark/runs/<run_id>/metrics/`
- `restapi/outputs/ontology_benchmark/runs/<run_id>/metrics/sparql_queries/`
- `summary.json` and `run_metadata.json`

## Bench4KE Validator UI

File name: `bench4ke-validate-ui.py`

Purpose: a small Flask UI with two tabs (CQ validation and ontology benchmark).

Run:

```bash
cd restapi
python bench4ke-validate-ui.py
```

Open `http://127.0.0.1:5000` and use the CQ Validation or Ontology Benchmark tabs.

## Optional: KG mapping generator

File name: `kg-generator.py`

Purpose: given a dataset and an ontology URI, prompts an LLM to produce an RML mapping and a SPARQL Anything script. This is not required for the ontology benchmark task but remains available.
