# === INSTALL ===

install:
	pip install -r requirements.txt

# === RUN ===

run-backend:
	uvicorn api.fast:app --reload --port 8000

run-frontend:
	streamlit run frontend/app.py

# === TRAINING ===

build-dataset:
	python model/build_dataset.py

train-model:
	python model/train_model.py

# === EVALUATE ===

evaluate-model:
	python model/evaluate_model.py

# === HELP ===

help:
	@echo "Usage:"
	@echo "  make install          Install all dependencies"
	@echo "  make run-backend      Run FastAPI backend (http://localhost:8000)"
	@echo "  make run-frontend     Run Streamlit frontend (http://localhost:8501)"
	@echo "  make build-dataset    Build image dataset from CSV"
	@echo "  make train-model      Train and save VGG16 model"
	@echo "  make evaluate-model   Evaluate model on test set"
