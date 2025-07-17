# === INSTALL ===

install:
	pip install -r requirements.txt

# === RUN ===

run-backend:
	uvicorn api.fast:app --reload --port 8000

run-frontend:
	streamlit run frontend/app.py

# === DEV ===

dev:
	@echo "Launching backend and frontend in parallel..."
	@echo "Press Ctrl+C to stop."
	# Requires 'concurrently' tool or use separate terminals manually
	# See docs or README for manual run

# === HELP ===

help:
	@echo "Usage:"
	@echo "  make install         Install all dependencies"
	@echo "  make run-backend     Run FastAPI backend (http://localhost:8000)"
	@echo "  make run-frontend    Run Streamlit frontend (http://localhost:8501)"
