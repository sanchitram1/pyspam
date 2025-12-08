from fastapi import FastAPI

app = FastAPI(title="PySpam API", description="API for PySpam")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "PySpam API is running"}
