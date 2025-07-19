# Art DNA

Machine learning project to predict art styles from paintings.

## Setup & Run

```bash
# Install dependencies
make install

# Create dataset (first time only)
make build-dataset

# Run backend (terminal 1)
make run-backend

# Run frontend (terminal 2)
make run-frontend
```

Open `http://localhost:8501` to upload images and predict art styles.

## Commands

```bash
make help          # Show all commands
make train-model   # Retrain model (delete existing model first)
```
