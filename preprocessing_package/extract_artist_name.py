import os
import pandas as pd
import re


def extract_artist_name_from_filename(filename):
    match = re.match(r"([A-Za-z_]+)_\d+\.jpg", filename)
    if match:
        return match.group(1).replace("_", " ")
    return None


def create_image_artist_df(images_root="raw_data/resized"):
    data = []
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.lower().endswith(".jpg"):
                artist_name = extract_artist_name_from_filename(file)
                rel_path = os.path.join(root, file)
                if artist_name:
                    data.append({"image_path": rel_path, "artist_name": artist_name})
    return pd.DataFrame(data)
