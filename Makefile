default:
	@echo "Please specify a target to run"

# === HELP ===

help:
	@echo "Usage:"
	@echo "  make install          Install all dependencies"
	@echo "  make run-backend      Run FastAPI backend with local model"
	@echo "  make run-backend-gcs  Run FastAPI backend with GCS model"
	@echo "  make run-frontend     Run Streamlit frontend"
	@echo "  make build-dataset    Build image dataset from CSV"
	@echo "  make train-model      Train and save VGG16 model"
	@echo "  make evaluate-model   Evaluate model on test set"

# === INSTALL ===

install:
	pip install -r requirements.txt

# === RUN ===

run-backend:
	uvicorn api.fast:app --reload --port 8000

run-backend-gcs:
	USE_GCS=true uvicorn api.fast:app --reload --port 8000

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

# === LOCAL DOCKER ===

build_container_local:
	docker build --tag=${IMAGE}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 ${IMAGE}:dev

# === GCP DEPLOYMENT ===

build_for_production:
	docker build \
		--platform linux/amd64 \
		-t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION}
