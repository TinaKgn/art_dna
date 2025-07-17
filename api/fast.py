from typing import Dict
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def dummy_predict(image_file: UploadFile) -> Dict[str, float]:
    return {"Impressionism": 0.7, "Cubism": 0.2, "Others": 0.1}


@app.get("/")
def root():
    return {"greeting": "Hello"}


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    predictions = dummy_predict(image)
    return {"predictions": predictions}
