from typing import Any, Dict

import requests


def run_oops_scan(
    ontology_text: str,
    api_url: str,
    timeout: float,
    mode: str = "text",
    ontology_url: str | None = None,
) -> Dict[str, Any]:
    if not api_url:
        return {"skipped": True, "reason": "OOPS_API_URL not set"}

    if mode == "url":
        if not ontology_url:
            return {"skipped": True, "reason": "OOPS_API_MODE=url requires ontology_url"}
        payload = {"ontologyURL": ontology_url}
        response = requests.post(api_url, data=payload, timeout=timeout)
    elif mode == "file":
        response = requests.post(
            api_url,
            files={"ontology": ("ontology.ttl", ontology_text)},
            timeout=timeout,
        )
    else:
        response = requests.post(
            api_url, data={"ontology": ontology_text}, timeout=timeout
        )

    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_response": response.text}
