import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

GENERATOR_CONFIG = {
    "target_size": (224, 224),  # size of images to process (default (256, 256))
    "batch_size": 32,  # number of images to process at once (default 32)
    "class_mode": "categorical",  # type of output (default categorical)
}

datagen = ImageDataGenerator(rescale=1.0 / 255)


def create_generator(dataframe, shuffle=True):
    return datagen.flow_from_dataframe(
        dataframe,
        directory="raw_data/resized",
        x_col="filename",
        y_col="genre",
        shuffle=shuffle,
        **GENERATOR_CONFIG,  # unpacks dict
    )


def get_data_generators(csv_path, include_test=False):
    df = pd.read_csv(csv_path)

    # First split on all data: 80% train+val / 20% test
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["genre"], random_state=42
    )

    # Second split on train+val: 80% train / 20% val
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df["genre"], random_state=42
    )

    train_gen = create_generator(train_df, shuffle=True)
    val_gen = create_generator(val_df, shuffle=True)

    if include_test:
        test_gen = create_generator(test_df, shuffle=False)
        return train_gen, val_gen, test_gen

    return train_gen, val_gen
