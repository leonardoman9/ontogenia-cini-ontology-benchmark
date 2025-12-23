import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _split_field(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    parts = [p.strip() for p in text.split("||")]
    return [p for p in parts if p]


def _get_first(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in row:
            val = _clean_str(row[key])
            if val:
                return val
    return ""


def build_ontology_request_payload(
    row: pd.Series,
    output_format: str = "turtle"
) -> Dict[str, Any]:
    domain_id = _get_first(row, ["domain_id", "project_name", "Project Name"])
    description = _get_first(row, ["description", "Description"])
    scenario = _get_first(row, ["scenario", "Scenario"])
    dataset_ref = _get_first(row, ["Dataset", "dataset_ref"])
    cqs_raw = _get_first(row, ["competency_questions", "Competency Question", "gold standard"])
    user_stories_raw = _get_first(row, ["user_stories", "User Stories", "User Stories (optional)"])
    constraints_raw = _get_first(row, ["constraints", "Constraints"])

    if not domain_id:
        raise ValueError("Missing domain_id in dataset row.")
    if not description and not scenario:
        raise ValueError("Missing description/scenario in dataset row.")

    competency_questions = _split_field(cqs_raw)
    if not competency_questions:
        raise ValueError("Missing competency_questions in dataset row.")

    user_stories = _split_field(user_stories_raw)

    constraints: Dict[str, Any] = {"output_format": output_format}
    if constraints_raw:
        parsed = None
        try:
            parsed = json.loads(constraints_raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            constraints.update(parsed)
        else:
            constraints["raw"] = constraints_raw

    payload: Dict[str, Any] = {
        "domain_id": domain_id,
        "competency_questions": competency_questions,
        "constraints": constraints,
    }
    if description:
        payload["description"] = description
    if scenario:
        payload["scenario"] = scenario
    if user_stories:
        payload["user_stories"] = user_stories
    if dataset_ref:
        payload["dataset_ref"] = dataset_ref

    return payload


def call_external_ontology_generation_service(
    payload: Dict[str, Any],
    external_service_url: str,
    timeout: int = 120
) -> Dict[str, Any]:
    try:
        response = requests.post(external_service_url, json=payload, timeout=timeout)
    except Exception as exc:
        raise Exception(f"Error calling external ontology service: {exc}") from exc

    if response.status_code != 200:
        raise Exception(f"External ontology service error: {response.text}")

    try:
        data = response.json()
    except Exception as exc:
        raise Exception(f"Invalid JSON response from ontology service: {exc}") from exc

    if "ontology" not in data:
        raise Exception("Ontology service did not return an 'ontology' field.")

    result = {
        "ontology": data["ontology"],
        "format": data.get("format", payload.get("constraints", {}).get("output_format", "turtle")),
        "actual_format": data.get("actual_format"),
        "warnings": data.get("warnings", []),
    }
    return result
