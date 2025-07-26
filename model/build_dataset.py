"""
Art DNA Dataset Builder

Processes the raw artists.csv and image files to create a labeled dataset for training.
Applies genre simplification mapping to reduce 24 original genres to 18 simplified categories
that align with the production VGG16 model and description system.

Outputs:
- raw_data/labeled_data.csv: Image filenames with corresponding simplified genres
- model/class_names.txt: List of 18 simplified genre names for model training

The genre simplification ensures consistency between:
- Model training pipeline
- API prediction responses
- Genre description system
"""

from pathlib import Path
import pandas as pd

artists_df = pd.read_csv("raw_data/artists.csv")

# Normalize artist names (lowercase, underscores for matching)
artists_df["normalized_name"] = artists_df["name"].str.lower().str.replace(" ", "_")

# Genre simplification mapping (24 -> 18 genres)
# This reduces the original complex genre combinations to 18 simplified categories
genre_mapping = {
    "Social Realism,Muralism": "Realism,Muralism",
    "Post-Impressionism": "Impressionism",
    "Northern Renaissance": "Renaissance",
    "Proto Renaissance": "Renaissance",
    "Early Renaissance": "Renaissance",
    "High Renaissance": "Renaissance",
    "Impressionism,Post-Impressionism": "Impressionism",
    "High Renaissance,Mannerism": "Renaissance,Mannerism",
    "Symbolism,Post-Impressionism": "Symbolism,Impressionism",
    "Abstract Expressionism": "Expressionism",
}


def simplify_genre(genre_string):
    """Apply genre simplification mapping to reduce 24 genres to 18"""
    return genre_mapping.get(genre_string, genre_string)


# Apply genre simplification, then normalize and split
artists_df["genre_simplified"] = artists_df["genre"].apply(simplify_genre)
artists_df["genre_simplified"] = (
    artists_df["genre_simplified"]
    .fillna("")
    .apply(lambda g: [s.strip() for s in g.split(",") if s.strip()])
)

# Map normalized artist names to simplified genres list
artist_to_genres = dict(
    zip(artists_df["normalized_name"], artists_df["genre_simplified"])
)

# Load image filenames
image_dir = Path("raw_data/resized")
image_files = list(image_dir.glob("*.jpg"))

# Build final dataset
rows = []
for path in image_files:
    filename = path.name
    # Normalize filename artist name
    ARTIST_NAME = "_".join(filename.split("_")[:-1]).lower()
    genres = artist_to_genres.get(ARTIST_NAME, [])
    for genre in genres:
        rows.append({"filename": filename, "genre": genre})

# Save result
df = pd.DataFrame(rows)
df.to_csv("raw_data/labeled_data.csv", index=False)

# Generate and save class names (18 simplified genres)
unique_genres = sorted(df["genre"].unique())
with open("model/class_names.txt", "w", encoding="utf-8") as f:
    for genre in unique_genres:
        f.write(f"{genre}\n")

# Summary
print(f"Saved {len(df)} rows to raw_data/labeled_data.csv")
print(f"Unique genres: {df['genre'].nunique()}")
print(f"Updated model/class_names.txt with {len(unique_genres)} simplified genres")
print("\nGenre mapping applied:")
for original, simplified in genre_mapping.items():
    print(f"  {original} → {simplified}")
print(f"\nFinal 18 genres: {unique_genres}")
print("Genre distribution:")
print(df["genre"].value_counts())
