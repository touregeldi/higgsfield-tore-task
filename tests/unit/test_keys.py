from memory_service.extraction.keys import normalize_value, KEY_LOCATION, KEY_EMPLOYMENT


def test_normalize_value_trims_and_collapses():
    assert normalize_value("  New   York  ") == "New York"


def test_normalize_value_strips_trailing_punctuation():
    assert normalize_value("Berlin.") == "Berlin"


def test_keys_are_stable_strings():
    assert KEY_LOCATION == "location"
    assert KEY_EMPLOYMENT == "employment"
