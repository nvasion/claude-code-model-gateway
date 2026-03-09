"""Configuration schema definitions for the model gateway.

Provides a typed schema for describing, documenting, and validating
gateway configuration fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FieldType(str, Enum):
    """Type of a schema field."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DICT = "dict"
    LIST = "list"
    OBJECT = "object"


class FieldCategory(str, Enum):
    """Category grouping for schema fields."""

    GATEWAY = "gateway"
    PROVIDER = "provider"
    MODEL = "model"
    LOGGING = "logging"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class FieldConstraint:
    """Value constraints for a schema field.

    Attributes:
        min_value: Minimum numeric value (inclusive).
        max_value: Maximum numeric value (inclusive).
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regex pattern a string must match.
        allowed_values: Explicit set of allowed values.
        required: Whether the field is required.
    """

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[list] = None
    required: bool = False

    def validate(self, value: Any) -> list[str]:
        """Validate a value against these constraints.

        Returns:
            List of error strings (empty if valid).
        """
        errors = []

        if value is None:
            return errors

        # Numeric range checks
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Value {value} is below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Value {value} exceeds maximum {self.max_value}")

        # String checks
        if isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(
                    f"String length {len(value)} is below minimum {self.min_length}"
                )
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(
                    f"String length {len(value)} exceeds maximum {self.max_length}"
                )
            if self.pattern is not None:
                if not re.match(self.pattern, value):
                    errors.append(f"Value does not match pattern '{self.pattern}'")

        # Allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            errors.append(
                f"Value {value!r} is not one of: {self.allowed_values}"
            )

        return errors


@dataclass
class SchemaField:
    """Description of a single configuration field.

    Attributes:
        name: Field name.
        field_type: The type of this field.
        description: Human-readable description.
        default: Default value.
        constraint: Optional value constraints.
        env_var: Environment variable that overrides this field.
        deprecated: Whether this field is deprecated.
        deprecation_message: Migration guidance.
        children: Child fields for OBJECT type.
        category: Category for grouping.
    """

    name: str
    field_type: FieldType
    description: str
    default: Any = None
    constraint: FieldConstraint = field(default_factory=FieldConstraint)
    env_var: Optional[str] = None
    deprecated: bool = False
    deprecation_message: str = ""
    children: dict[str, "SchemaField"] = field(default_factory=dict)
    category: FieldCategory = FieldCategory.GATEWAY

    def validate_value(self, value: Any) -> list[str]:
        """Validate a value against this field's type and constraints.

        Returns:
            List of error strings (empty if valid).
        """
        errors = []

        if value is None:
            return errors  # None skips type checking

        # Type checking
        if self.field_type == FieldType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    f"Field '{self.name}' must be an integer, got {type(value).__name__}"
                )
        elif self.field_type == FieldType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(
                    f"Field '{self.name}' must be a float, got {type(value).__name__}"
                )
        elif self.field_type == FieldType.STRING:
            if not isinstance(value, str):
                errors.append(
                    f"Field '{self.name}' must be a string, got {type(value).__name__}"
                )
        elif self.field_type == FieldType.BOOLEAN:
            if not isinstance(value, bool):
                errors.append(
                    f"Field '{self.name}' must be a boolean, got {type(value).__name__}"
                )
        elif self.field_type == FieldType.ENUM:
            if not isinstance(value, str):
                errors.append(
                    f"Field '{self.name}' must be a string (enum), got {type(value).__name__}"
                )
        elif self.field_type == FieldType.DICT:
            if not isinstance(value, dict):
                errors.append(
                    f"Field '{self.name}' must be a dict, got {type(value).__name__}"
                )
        elif self.field_type == FieldType.LIST:
            if not isinstance(value, list):
                errors.append(
                    f"Field '{self.name}' must be a list, got {type(value).__name__}"
                )

        # Constraint checks (only if no type error)
        if not errors and self.constraint:
            errors.extend(self.constraint.validate(value))

        return errors

    def get_child(self, name: str) -> Optional["SchemaField"]:
        """Get a child field by name."""
        return self.children.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for documentation."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.field_type.value,
            "description": self.description,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.env_var:
            d["env_var"] = self.env_var
        if self.deprecated:
            d["deprecated"] = True
            d["deprecation_message"] = self.deprecation_message
        if self.constraint:
            if self.constraint.allowed_values:
                d["allowed_values"] = self.constraint.allowed_values
            if self.constraint.min_value is not None:
                d["min_value"] = self.constraint.min_value
            if self.constraint.max_value is not None:
                d["max_value"] = self.constraint.max_value
            if self.constraint.required:
                d["required"] = True
        if self.children:
            d["children"] = {k: v.to_dict() for k, v in self.children.items()}
        return d


# ---------------------------------------------------------------------------
# Built-in provider schema
# ---------------------------------------------------------------------------

MODEL_SCHEMA = SchemaField(
    name="model",
    field_type=FieldType.OBJECT,
    description="Model configuration.",
    category=FieldCategory.MODEL,
    children={
        "name": SchemaField(
            name="name",
            field_type=FieldType.STRING,
            description="Model identifier.",
            category=FieldCategory.MODEL,
            constraint=FieldConstraint(required=True),
        ),
        "display_name": SchemaField(
            name="display_name",
            field_type=FieldType.STRING,
            description="Human-readable model name.",
            category=FieldCategory.MODEL,
        ),
        "max_tokens": SchemaField(
            name="max_tokens",
            field_type=FieldType.INTEGER,
            description="Maximum token limit for this model.",
            default=4096,
            category=FieldCategory.MODEL,
            constraint=FieldConstraint(min_value=1, max_value=10_000_000),
        ),
        "supports_streaming": SchemaField(
            name="supports_streaming",
            field_type=FieldType.BOOLEAN,
            description="Whether the model supports streaming responses.",
            default=True,
            category=FieldCategory.MODEL,
        ),
        "supports_tools": SchemaField(
            name="supports_tools",
            field_type=FieldType.BOOLEAN,
            description="Whether the model supports tool/function calling.",
            default=False,
            category=FieldCategory.MODEL,
        ),
        "supports_vision": SchemaField(
            name="supports_vision",
            field_type=FieldType.BOOLEAN,
            description="Whether the model supports vision/image inputs.",
            default=False,
            category=FieldCategory.MODEL,
        ),
    },
)

PROVIDER_SCHEMA = SchemaField(
    name="provider",
    field_type=FieldType.OBJECT,
    description="Provider configuration.",
    category=FieldCategory.PROVIDER,
    children={
        "name": SchemaField(
            name="name",
            field_type=FieldType.STRING,
            description="Unique provider identifier.",
            category=FieldCategory.PROVIDER,
            constraint=FieldConstraint(required=True),
        ),
        "display_name": SchemaField(
            name="display_name",
            field_type=FieldType.STRING,
            description="Human-readable provider name.",
            category=FieldCategory.PROVIDER,
        ),
        "api_base": SchemaField(
            name="api_base",
            field_type=FieldType.STRING,
            description="Base URL for the provider's API.",
            category=FieldCategory.PROVIDER,
            constraint=FieldConstraint(required=True),
        ),
        "api_key_env_var": SchemaField(
            name="api_key_env_var",
            field_type=FieldType.STRING,
            description="Environment variable containing the API key.",
            category=FieldCategory.PROVIDER,
        ),
        "auth_type": SchemaField(
            name="auth_type",
            field_type=FieldType.ENUM,
            description="Authentication method.",
            default="api_key",
            category=FieldCategory.PROVIDER,
            constraint=FieldConstraint(
                allowed_values=["api_key", "bearer_token", "none"]
            ),
        ),
        "default_model": SchemaField(
            name="default_model",
            field_type=FieldType.STRING,
            description="Default model name for this provider.",
            category=FieldCategory.PROVIDER,
        ),
        "enabled": SchemaField(
            name="enabled",
            field_type=FieldType.BOOLEAN,
            description="Whether this provider is currently enabled.",
            default=True,
            category=FieldCategory.PROVIDER,
        ),
        "models": SchemaField(
            name="models",
            field_type=FieldType.OBJECT,
            description="Available models for this provider.",
            category=FieldCategory.PROVIDER,
            children={"*": MODEL_SCHEMA},
        ),
    },
)

GATEWAY_SCHEMA = SchemaField(
    name="gateway",
    field_type=FieldType.OBJECT,
    description="Top-level gateway configuration.",
    category=FieldCategory.GATEWAY,
    children={
        "default_provider": SchemaField(
            name="default_provider",
            field_type=FieldType.STRING,
            description="Name of the default provider.",
            default="",
            category=FieldCategory.GATEWAY,
        ),
        "log_level": SchemaField(
            name="log_level",
            field_type=FieldType.ENUM,
            description="Logging verbosity level.",
            default="info",
            env_var="GATEWAY_LOG_LEVEL",
            category=FieldCategory.LOGGING,
            constraint=FieldConstraint(
                allowed_values=["debug", "info", "warning", "error", "critical"]
            ),
        ),
        "timeout": SchemaField(
            name="timeout",
            field_type=FieldType.INTEGER,
            description="Default request timeout in seconds.",
            default=30,
            env_var="GATEWAY_TIMEOUT",
            category=FieldCategory.PERFORMANCE,
            constraint=FieldConstraint(min_value=1, max_value=3600),
        ),
        "max_retries": SchemaField(
            name="max_retries",
            field_type=FieldType.INTEGER,
            description="Maximum number of retry attempts.",
            default=3,
            env_var="GATEWAY_MAX_RETRIES",
            category=FieldCategory.PERFORMANCE,
            constraint=FieldConstraint(min_value=0, max_value=100),
        ),
        "providers": SchemaField(
            name="providers",
            field_type=FieldType.OBJECT,
            description="Dictionary of provider configurations.",
            category=FieldCategory.PROVIDER,
            children={"*": PROVIDER_SCHEMA},
        ),
    },
)


# ---------------------------------------------------------------------------
# Schema helper functions
# ---------------------------------------------------------------------------


def get_field(
    path: str, schema: Optional[SchemaField] = None
) -> Optional[SchemaField]:
    """Get a schema field by dotted path (supports * wildcards).

    Args:
        path: Dotted path like 'timeout' or 'providers.*.api_base'.
        schema: Schema to search in. Defaults to GATEWAY_SCHEMA.

    Returns:
        The SchemaField at the given path, or None if not found.
    """
    if schema is None:
        schema = GATEWAY_SCHEMA

    parts = path.split(".")
    current = schema

    for part in parts:
        if not current.children:
            return None
        if part in current.children:
            current = current.children[part]
        elif "*" in current.children:
            current = current.children["*"]
            # If part is not a literal child of the wildcard's children,
            # look in the wildcard field's children for this part
            if part not in current.children and "*" not in current.children:
                return None
            # If the current part is a key inside the wildcard schema's children
            # we don't advance again - we've already moved to the wildcard schema
        else:
            return None

    return current


def _collect_fields(
    schema: SchemaField,
    category: Optional[FieldCategory],
    include_nested: bool,
    result: list[SchemaField],
    _depth: int = 0,
) -> None:
    """Recursively collect schema fields."""
    if _depth > 0:  # Don't include the root schema itself
        if category is None or schema.category == category:
            result.append(schema)
    if include_nested or _depth == 0:
        for child in schema.children.values():
            if child.name == "*":
                # Expand wildcard children
                _collect_fields(child, category, include_nested, result, _depth + 1)
            else:
                _collect_fields(child, category, include_nested, result, _depth + 1)


def list_fields(
    category: Optional[FieldCategory] = None,
    include_nested: bool = True,
    schema: Optional[SchemaField] = None,
) -> list[SchemaField]:
    """List schema fields, optionally filtered by category.

    Args:
        category: Optional category filter.
        include_nested: Whether to include nested fields (provider/model).
        schema: Schema to list from. Defaults to GATEWAY_SCHEMA.

    Returns:
        List of SchemaField objects.
    """
    if schema is None:
        schema = GATEWAY_SCHEMA
    result: list[SchemaField] = []
    _collect_fields(schema, category, include_nested, result)
    return result


def get_defaults(schema: Optional[SchemaField] = None) -> dict[str, Any]:
    """Get all default values from the schema.

    Args:
        schema: Schema to extract defaults from. Defaults to GATEWAY_SCHEMA.

    Returns:
        Dict mapping field name to default value.
    """
    if schema is None:
        schema = GATEWAY_SCHEMA

    defaults: dict[str, Any] = {}

    def _collect(s: SchemaField, prefix: str = "") -> None:
        if s.default is not None:
            key = f"{prefix}{s.name}" if prefix else s.name
            defaults[key] = s.default
        for child in s.children.values():
            if child.name != "*":
                _collect(child, "")

    _collect(schema)
    return defaults


def validate_against_schema(
    data: dict[str, Any],
    schema: Optional[SchemaField] = None,
) -> list[str]:
    """Validate a configuration dict against a schema.

    Args:
        data: Configuration dictionary to validate.
        schema: Schema to validate against. Defaults to GATEWAY_SCHEMA.

    Returns:
        List of error strings (empty if valid).
    """
    if schema is None:
        schema = GATEWAY_SCHEMA

    errors: list[str] = []
    _validate_dict(data, schema, "", errors)
    return errors


def _validate_dict(
    data: dict[str, Any],
    schema: SchemaField,
    path_prefix: str,
    errors: list[str],
) -> None:
    """Recursively validate a dict against a schema."""
    for key, value in data.items():
        field_schema = schema.children.get(key) or schema.children.get("*")
        if field_schema is None:
            continue  # Unknown fields are ignored

        field_path = f"{path_prefix}.{key}" if path_prefix else key

        if field_schema.name == "*":
            # Wildcard: value should be a dict of items matching the wildcard schema
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_path = f"{field_path}.{sub_key}"
                    if isinstance(sub_value, dict):
                        _validate_dict(sub_value, field_schema, sub_path, errors)
                    else:
                        errs = field_schema.validate_value(sub_value)
                        errors.extend(f"{sub_path}: {e}" for e in errs)
            continue

        # Type and constraint validation
        errs = field_schema.validate_value(value)
        errors.extend(f"{field_path}: {e}" for e in errs)

        # Recurse into dicts for OBJECT fields
        if (
            field_schema.field_type == FieldType.OBJECT
            and isinstance(value, dict)
            and field_schema.children
        ):
            _validate_dict(value, field_schema, field_path, errors)


def generate_documented_config() -> str:
    """Generate a documented YAML template showing all configuration options.

    Returns:
        A YAML string with inline comments documenting each field.
    """
    lines = [
        "# claude-code-model-gateway Configuration",
        "# Generated documentation of all available configuration options.",
        "#",
    ]

    def _add_field(f: SchemaField, indent: int = 0) -> None:
        prefix = "  " * indent

        if f.name in ("*", "gateway"):
            return

        # Comment block
        lines.append(f"{prefix}# {f.description}")
        if f.env_var:
            lines.append(f"{prefix}# Env override: {f.env_var}")
        if f.constraint and f.constraint.allowed_values:
            vals = ", ".join(str(v) for v in f.constraint.allowed_values)
            lines.append(f"{prefix}# Allowed values: {vals}")
        if f.constraint and f.constraint.min_value is not None:
            lines.append(f"{prefix}# Min: {f.constraint.min_value}")
        if f.constraint and f.constraint.max_value is not None:
            lines.append(f"{prefix}# Max: {f.constraint.max_value}")
        if f.deprecated:
            lines.append(f"{prefix}# DEPRECATED: {f.deprecation_message}")

        if f.default is not None:
            lines.append(f"{prefix}{f.name}: {f.default}")
        elif f.field_type == FieldType.OBJECT:
            lines.append(f"{prefix}{f.name}:")
            for child in f.children.values():
                if child.name != "*":
                    _add_field(child, indent + 1)
        else:
            lines.append(f"{prefix}{f.name}:  # {f.field_type.value}")

        lines.append("")

    for child in GATEWAY_SCHEMA.children.values():
        _add_field(child)

    lines.append("")
    return "\n".join(lines) + "\n"
