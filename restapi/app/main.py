# app/main.py
from fastapi import FastAPI
from app.routers import cq_validation, ontology_benchmark

app = FastAPI(
    title="Bench4KE API",
    description="APIs for CQ validation and ontology generation benchmarking",
    version="1.0.0"
)

# Include routers with prefixes and tags
app.include_router(cq_validation.router, prefix="/validate", tags=["CQ Validation"])
app.include_router(ontology_benchmark.router, prefix="/benchmark/ontology", tags=["Ontology Benchmark"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
