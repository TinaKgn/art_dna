import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(csv_path):
    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["genre"], random_state=42
    )

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory="raw_data/resized",
        x_col="filename",
        y_col="genre",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    val_gen = datagen.flow_from_dataframe(
        val_df,
        directory="raw_data/resized",
        x_col="filename",
        y_col="genre",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    return train_gen, val_gen
