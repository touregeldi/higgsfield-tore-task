import re

KEY_LOCATION = "location"
KEY_EMPLOYMENT = "employment"
KEY_ROLE = "role"
KEY_PET_NAME = "pet.name"
KEY_FAMILY = "family"

_WS = re.compile(r"\s+")


def normalize_value(raw: str) -> str:
    v = _WS.sub(" ", raw.strip())
    return v.rstrip(".!,;: ").strip()


def pet_key(name: str) -> str:
    return f"{KEY_PET_NAME}:{normalize_value(name).lower()}"
