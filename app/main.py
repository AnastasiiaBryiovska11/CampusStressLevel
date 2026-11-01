from fastapi import FastAPI
import uvicorn

from app.endpoints.training import router as training_router
from app.endpoints.inference import router as inference_router


app = FastAPI()

app.include_router(training_router)
app.include_router(inference_router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )