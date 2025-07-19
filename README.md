# Art DNA

Machine learning project to predict art styles from paintings.

## Setup & Run

```bash
# Install dependencies
make install

# Create dataset (first time only)
make build-dataset

# Train model (first time only)
make train-model

# Evaluate model performance on test set
make evaluate-model

# Run backend (terminal 1)
make run-backend

# Run frontend (terminal 2)
make run-frontend
```

Open `http://localhost:8501` to upload images and predict art styles.

## Commands

```bash
make help            # Show all commands
make build-dataset   # Build image dataset from CSV
make train-model     # Train and save VGG16 model
make evaluate-model  # Evaluate model on test set
make run-backend     # Run FastAPI backend
make run-frontend    # Run Streamlit frontend
```
