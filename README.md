# Art DNA

Machine learning project to classify art styles from paintings using CNN transfer learning.

## Overview

- **Classification**: 18 art genres with probabilistic multi-label output
- **Model**: VGG16 fine-tuned with transfer learning
- **Descriptions**: Rich educational content for both adult and kid audiences
- **API**: FastAPI backend with image upload and genre descriptions

## Prerequisites

- Python 3.10+
- Download [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data) dataset

## Setup

```bash
# Install dependencies
make install

# Extract dataset to raw_data/ directory
# Create dataset (first time only)
make build-dataset
```

## Run

```bash
# Run backend (terminal 1)
make run-backend

# Run frontend (terminal 2)
make run-frontend
```

Open `http://localhost:8501` to upload images and predict art styles.

## API

- `POST /predict` - Upload image, returns probabilities for 18 art genres
- `GET /describe` - Get educational descriptions for genres (adult/kid audience)
- Live demo: [art-dna-api.run.app](https://art-dna-api-521843227251.europe-west1.run.app)

## Train & Evaluate

```bash
# Train model (first time only, ~10-15 minutes)
make train-model

# Evaluate model performance on test set
make evaluate-model
```

## Deployment

```bash
# Docker
make build_container_local
make run_container_local

# GCP Cloud Run
make build_for_production
make push_image_production
make deploy_to_cloud_run
```

## Commands

```bash
make help            # Show all commands
make build-dataset   # Build image dataset with 18 simplified genres
make train-model     # Train and save VGG16 model
make evaluate-model  # Evaluate model on test set
make run-backend     # Run FastAPI backend
make run-frontend    # Run Streamlit frontend
```

## Team

- [Adriana]()
- [Anna](https://github.com/AnnaShe78)
- [Kristina](https://github.com/TinaKgn)
- [Marc](https://github.com/MarcRenard)
- [Michael](https://github.com/michaelsmueller)
