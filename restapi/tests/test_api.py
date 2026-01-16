import io
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

@pytest.mark.parametrize("use_default_dataset, file_upload", [
    (True, None),  # Test default dataset mode
    (False, io.BytesIO(b'gold standard,SomeOtherColumn\n'
                        b'"What is the project about?","ExtraValue1"\n'
                        b'"How many components are there?","ExtraValue2"\n'))  # Test file upload mode
])
def test_validate_competency_questions(use_default_dataset, file_upload, monkeypatch):
    import app.routers.cq_validation as cq_validation
    import app.services.cq_validator as cq_validator

    def stub_cq_service(df, external_service_url):
        df = df.copy()
        source_col = "gold standard" if "gold standard" in df.columns else df.columns[0]
        df["generated"] = df[source_col].apply(lambda val: f"Generated: {val}")
        return df

    class DummyChoice:
        def __init__(self, content):
            self.message = type("obj", (), {"content": content})

    class DummyResponse:
        def __init__(self, content):
            self.choices = [DummyChoice(content)]

    def stub_openai_create(*args, **kwargs):
        return DummyResponse("Stub analysis.")

    monkeypatch.setattr(cq_validation, "call_external_cq_generation_service", stub_cq_service)
    monkeypatch.setattr(cq_validator.openai.chat.completions, "create", stub_openai_create)

    data = {
        "validation_mode": "llm",
        "use_default_dataset": str(use_default_dataset),
        "external_service_url": "http://127.0.0.1:8001/newapi",
        "api_key": "key",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "save_results": "False"
    }
    files = {}
    if file_upload:
        files = {"file": ("benchmarkdataset.csv", file_upload, "text/csv")}
    response = client.post("/validate/", data=data, files=files)
    assert response.status_code == 200
    json_data = response.json()
    import json
    print(json.dumps(response.json(), indent=2))

    print("Response JSON:", response.json())
    assert "validation_results" in json_data
