"""
Art DNA Genre Descriptions

Loads genre descriptions from data files and provides them in the format expected by the API.
Supports both adult and kid-friendly descriptions for all 18 art genres.
"""

import json
from typing import Dict, Any

# Import data directly from Python files (no CSV parsing needed)
from data.adult_data import adult_data
from data.kids_data import kids_data


def _parse_json_field(json_string: str) -> list:
    """Parse JSON string field into Python list"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return []


def _build_descriptions_dict(data_list: list) -> Dict[str, Dict[str, Any]]:
    """Transform list format to nested dict format for API"""
    descriptions = {}

    for item in data_list:
        genre_name = item["genre"]
        descriptions[genre_name] = {
            "genre": genre_name,
            "description": item["description"],
            "time_period": item["time_period"],
            "key_artists": _parse_json_field(item["key_artists"]),
            "visual_elements": _parse_json_field(item["visual_elements"]),
            "philosophy": item["philosophy"],
        }

    return descriptions


# Build descriptions dict at module load time (startup performance)
DESCRIPTIONS = {
    "adult": _build_descriptions_dict(adult_data),
    "kid": _build_descriptions_dict(kids_data),
}

# Verify we have all 18 genres
EXPECTED_GENRES = 18
ADULT_COUNT = len(DESCRIPTIONS["adult"])
KIDS_COUNT = len(DESCRIPTIONS["kid"])

if ADULT_COUNT != EXPECTED_GENRES or KIDS_COUNT != EXPECTED_GENRES:
    print(
        f"⚠️  Warning: Expected {EXPECTED_GENRES} genres, got adult={ADULT_COUNT}, kid={KIDS_COUNT}"
    )
else:
    print(
        f"✅ Loaded {EXPECTED_GENRES} genre descriptions for both adult and kid audiences"
    )
