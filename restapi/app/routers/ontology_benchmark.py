import os
import re
import time
from io import StringIO

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import (
    DEFAULT_ONTOLOGY_DATASET,
    DEFAULT_ONTOLOGY_OUTPUT_FORMAT,
    ONTOLOGY_RESULTS_DIR,
    ONTOLOGY_RUNS_DIR,
)
from app.utils.external_ontology_call import (
    build_ontology_request_payload,
    call_external_ontology_generation_service,
)

router = APIRouter()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("_.")
    return cleaned or "ontology"


def _format_to_ext(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt in {"turtle", "ttl"}:
        return "ttl"
    if fmt in {"rdfxml", "rdf", "xml", "rdf/xml", "rdf-xml"}:
        return "rdf"
    if fmt in {"jsonld", "json-ld"}:
        return "jsonld"
    return "ttl"


def _row_domain_id(row: pd.Series, fallback: str) -> str:
    for key in ("domain_id", "project_name", "Project Name"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return fallback


@router.post("/")
async def benchmark_ontology_generation(
    file: UploadFile = File(None),
    use_default_dataset: bool = Form(False),
    external_service_url: str = Form(...),
    output_format: str = Form(DEFAULT_ONTOLOGY_OUTPUT_FORMAT),
    output_folder: str = Form(ONTOLOGY_RESULTS_DIR),
    results_folder: str = Form(ONTOLOGY_RUNS_DIR),
    save_results: bool = Form(True),
):
    if file is not None:
        try:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {exc}")
    elif use_default_dataset:
        try:
            df = pd.read_csv(DEFAULT_ONTOLOGY_DATASET)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error loading default dataset: {exc}")
    else:
        raise HTTPException(status_code=400, detail="Either upload a CSV file or set use_default_dataset=True.")

    os.makedirs(output_folder, exist_ok=True)
    if save_results:
        os.makedirs(results_folder, exist_ok=True)

    results = []
    for idx, row in df.iterrows():
        domain_id = _row_domain_id(row, f"row_{idx}")
        try:
            payload = build_ontology_request_payload(row, output_format=output_format)
            response_data = call_external_ontology_generation_service(payload, external_service_url)

            ontology_text = response_data["ontology"]
            fmt = response_data.get("format") or output_format
            actual_format = response_data.get("actual_format") or fmt

            ext = _format_to_ext(actual_format)
            filename = f"{_safe_filename(domain_id)}.{ext}"
            file_path = os.path.join(output_folder, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(ontology_text)

            results.append(
                {
                    "domain_id": domain_id,
                    "ontology_path": file_path,
                    "format": fmt,
                    "actual_format": actual_format,
                    "warnings": "; ".join(response_data.get("warnings", [])),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "domain_id": domain_id,
                    "error": str(exc),
                }
            )

    results_file = ""
    if save_results:
        timestamp = int(time.time())
        results_file = os.path.join(results_folder, f"ontology_benchmark_{timestamp}.csv")

        keys = set()
        for r in results:
            keys.update(r.keys())
        keys = list(keys)

        import csv

        with open(results_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    return JSONResponse(
        content={
            "message": "Processing complete",
            "results_saved_to": results_file if save_results else "Not saved",
            "benchmark_results": results,
        }
    )
