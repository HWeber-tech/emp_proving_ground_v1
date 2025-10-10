import json
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path("interfaces/rim_types.json")
EXAMPLE_PATH = Path("docs/examples/rim_suggestion_examples.jsonl")


def load_schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_examples_match_schema():
    schema_doc = load_schema()
    validator = jsonschema.Draft7Validator(
        schema_doc["definitions"]["RIMSuggestion"],
        resolver=jsonschema.RefResolver.from_schema(schema_doc),
    )
    lines = EXAMPLE_PATH.read_text().strip().splitlines()
    assert lines, "example JSONL must not be empty"
    for line in lines:
        payload = json.loads(line)
        validator.validate(payload)
        for field in ("schema_version", "input_hash", "model_hash", "config_hash"):
            assert field in payload and payload[field], f"missing required field {field}"
