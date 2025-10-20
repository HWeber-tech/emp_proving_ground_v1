"""Generate a Markdown registry describing the sensory organs."""

from __future__ import annotations

import argparse
import importlib
import inspect
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

__all__ = [
    "SensorDefinition",
    "ConfigField",
    "SensorRegistryEntry",
    "build_registry",
    "format_markdown",
    "format_json",
    "main",
]


@dataclass(frozen=True)
class SensorDefinition:
    """Description of a sensory organ implementation."""

    dimension: str
    module: str
    class_name: str
    config_name: str | None = None


@dataclass(frozen=True)
class ConfigField:
    """Configuration metadata extracted from a dataclass."""

    name: str
    type: str
    default: str


@dataclass(frozen=True)
class SensorRegistryEntry:
    """Materialised registry entry for a sensory organ."""

    dimension: str
    qualified_name: str
    description: str
    config_description: str | None
    config_fields: tuple[ConfigField, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "qualified_name": self.qualified_name,
            "description": self.description,
            "config_description": self.config_description,
            "config_fields": [field.__dict__ for field in self.config_fields],
        }


_DEFAULT_DEFINITIONS: tuple[SensorDefinition, ...] = (
    SensorDefinition(
        dimension="HOW",
        module="sensory.how.how_sensor",
        class_name="HowSensor",
        config_name="HowSensorConfig",
    ),
    SensorDefinition(
        dimension="WHAT",
        module="sensory.what.what_sensor",
        class_name="WhatSensor",
    ),
    SensorDefinition(
        dimension="WHEN",
        module="sensory.when.when_sensor",
        class_name="WhenSensor",
        config_name="WhenSensorConfig",
    ),
    SensorDefinition(
        dimension="WHY",
        module="sensory.why.why_sensor",
        class_name="WhySensor",
    ),
    SensorDefinition(
        dimension="ANOMALY",
        module="sensory.anomaly.anomaly_sensor",
        class_name="AnomalySensor",
        config_name="AnomalySensorConfig",
    ),
    SensorDefinition(
        dimension="CORRELATION",
        module="sensory.correlation.cross_market_correlation_sensor",
        class_name="CrossMarketCorrelationSensor",
    ),
)


def build_registry(
    definitions: Sequence[SensorDefinition] | None = None,
) -> list[SensorRegistryEntry]:
    """Return registry entries for the configured sensory organs."""

    definitions = list(definitions or _DEFAULT_DEFINITIONS)
    entries: list[SensorRegistryEntry] = []

    for definition in definitions:
        module = importlib.import_module(definition.module)
        sensor_cls = getattr(module, definition.class_name)
        description = inspect.getdoc(sensor_cls) or ""
        qualified_name = f"{definition.module}.{definition.class_name}"

        config_fields: list[ConfigField] = []
        config_description: str | None = None

        if definition.config_name:
            config_obj = getattr(module, definition.config_name)
            config_description = inspect.getdoc(config_obj)
            if is_dataclass(config_obj):
                for field in fields(config_obj):
                    config_fields.append(
                        ConfigField(
                            name=field.name,
                            type=_format_annotation(field.type),
                            default=_format_default(field),
                        )
                    )

        entries.append(
            SensorRegistryEntry(
                dimension=definition.dimension,
                qualified_name=qualified_name,
                description=description,
                config_description=config_description,
                config_fields=tuple(config_fields),
            )
        )

    return entries


def _format_annotation(annotation: object) -> str:
    if annotation is inspect._empty or annotation is None:
        return "Any"
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        formatted_args = ", ".join(_format_annotation(arg) for arg in args)
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{formatted_args}]"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    text = str(annotation)
    if text.startswith("typing."):
        text = text[len("typing.") :]
    return text


def _format_default(field) -> str:
    default = field.default
    if default is not MISSING:
        return _repr_default(default)
    factory = getattr(field, "default_factory", MISSING)
    if factory is not MISSING:
        if getattr(factory, "__name__", None):
            return f"factory:{factory.__name__}()"
        return "factory"  # pragma: no cover - best effort only
    return "required"


def _repr_default(value: object) -> str:
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    if value is None:
        return "None"
    return repr(value)


def format_markdown(entries: Iterable[SensorRegistryEntry]) -> str:
    """Return a Markdown document describing the sensory registry."""

    lines = [
        "# Sensory registry",
        "",
        "This registry summarises the high-impact sensory organs and their",
        "configuration surfaces. Regenerate via `python -m tools.sensory.registry`.",
        "",
    ]

    for entry in entries:
        lines.extend(
            [
                f"## {entry.dimension} – {entry.qualified_name}",
                "",
            ]
        )
        description = entry.description or "No class documentation available."
        lines.append(description)
        lines.append("")

        if entry.config_fields:
            lines.append("### Configuration")
            lines.append("")
            if entry.config_description:
                lines.append(entry.config_description)
                lines.append("")
            lines.append("| Field | Type | Default |")
            lines.append("| --- | --- | --- |")
            for field in entry.config_fields:
                lines.append(
                    f"| `{field.name}` | {field.type} | {field.default} |"
                )
            lines.append("")
        else:
            lines.append("*This sensor does not expose configuration parameters.*")
            lines.append("")

    return "\n".join(lines).rstrip()


def format_json(entries: Iterable[SensorRegistryEntry]) -> str:
    """Serialise entries as a JSON payload."""

    import json

    return json.dumps([entry.as_dict() for entry in entries], indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the sensory registry CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the registry should be written",
    )
    args = parser.parse_args(argv)

    entries = build_registry()
    if args.format == "json":
        content = format_json(entries)
    else:
        content = format_markdown(entries)

    if args.output:
        to_write = content if content.endswith("\n") else f"{content}\n"
        args.output.write_text(to_write, encoding="utf-8")

    print(content)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
