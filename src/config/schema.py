"""Configuration schema definitions for claude-code-model-gateway.

Provides a declarative schema describing every configuration field,
its type, constraints, defaults, and documentation.  The schema is
used by the validator, documentation generators, and the ``config init``
command to produce well-documented configuration files.

Example usage::

    from src.config.schema import GATEWAY_SCHEMA, get_field, list_fields

    field = get_field("timeout")
    print(field.description)  # "Default request timeout in seconds."
    print(field.default)      # 30
    print(field.field_type)   # "integer"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema field types
# ---------------------------------------------------------------------------


class FieldType(str, Enum):
    """Supported configuration field types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DICT = "dict"
    LIST = "list"
    OBJECT = "object"


class FieldCategory(str, Enum):
    """Logical grouping of configuration fields."""

    GATEWAY = "gateway"
    PROVIDER = "provider"
    MODEL = "model"
    LOGGING = "logging"
    RETRY = "retry"
    PROXY = "proxy"


# ---------------------------------------------------------------------------
# Schema field definition
# ---------------------------------------------------------------------------


@dataclass
class FieldConstraint:
    """Value constraint for a schema field.

    Attributes:
        min_value: Minimum numeric value (inclusive).
        max_value: Maximum numeric value (inclusive).
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regex pattern the value must match.
        allowed_values: List of allowed enum values.
        required: Whether the field is required.
        unique: Whether the value must be unique among siblings.
    """

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: list[str] = field(default_factory=list)
    required: bool = False
    unique: bool = False

    def validate(self, value: Any) -> list[str]:
        """Validate a value against this constraint.

        Args:
            value: The value to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                errors.append(
                    f"Value {value} is below minimum {self.min_value}."
                )

        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                errors.append(
                    f"Value {value} exceeds maximum {self.max_value}."
                )

        if self.min_length is not None and isinstance(value, str):
            if len(value) < self.min_length:
                errors.append(
                    f"Length {len(value)} is below minimum {self.min_length}."
                )

        if self.max_length is not None and isinstance(value, str):
            if len(value) > self.max_length:
                errors.append(
                    f"Length {len(value)} exceeds maximum {self.max_length}."
                )

        if self.pattern is not None and isinstance(value, str):
            import re

            if not re.match(self.pattern, value):
                errors.append(
                    f"Value '{value}' does not match pattern '{self.pattern}'."
                )

        if self.allowed_values and isinstance(value, str):
            if value not in self.allowed_values:
                errors.append(
                    f"Value '{value}' is not one of: "
                    f"{', '.join(self.allowed_values)}."
                )

        return errors


@dataclass
class SchemaField:
    """Definition of a single configuration field.

    Attributes:
        name: Field name (dotted path for nested fields).
        field_type: Data type of the field.
        description: Human-readable description.
        default: Default value when not specified.
        category: Logical category this field belongs to.
        constraint: Value constraints for validation.
        children: Child fields for object/dict types.
        deprecated: Whether this field is deprecated.
        deprecation_message: Message shown when deprecated field is used.
        examples: Example values for documentation.
        env_var: Environment variable that can override this field.
    """

    name: str
    field_type: FieldType
    description: str
    default: Any = None
    category: FieldCategory = FieldCategory.GATEWAY
    constraint: FieldConstraint = field(default_factory=FieldConstraint)
    children: dict[str, "SchemaField"] = field(default_factory=dict)
    deprecated: bool = False
    deprecation_message: str = ""
    examples: list[Any] = field(default_factory=list)
    env_var: Optional[str] = None

    def get_child(self, name: str) -> Optional["SchemaField"]:
        """Look up a direct child field by name.

        Args:
            name: The child field name.

        Returns:
            The child SchemaField if found, None otherwise.
        """
        return self.children.get(name)

    def validate_value(self, value: Any) -> list[str]:
        """Validate a value against this field's type and constraints.

        Args:
            value: The value to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Type check
        expected_types = _FIELD_TYPE_MAP.get(self.field_type)
        if expected_types and value is not None:
            if not isinstance(value, expected_types):
                errors.append(
                    f"Expected {self.field_type.value}, "
                    f"got {type(value).__name__}."
                )
                return errors  # Skip constraint checks on wrong type

        # Constraint check
        if self.constraint:
            errors.extend(self.constraint.validate(value))

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the field definition to a dictionary.

        Returns:
            Dictionary representation of this schema field.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.field_type.value,
            "description": self.description,
        }
        if self.default is not None:
            data["default"] = self.default
        if self.category:
            data["category"] = self.category.value
        if self.constraint.required:
            data["required"] = True
        if self.constraint.allowed_values:
            data["allowed_values"] = self.constraint.allowed_values
        if self.env_var:
            data["env_var"] = self.env_var
        if self.deprecated:
            data["deprecated"] = True
            if self.deprecation_message:
                data["deprecation_message"] = self.deprecation_message
        if self.examples:
            data["examples"] = self.examples
        if self.children:
            data["children"] = {
                k: v.to_dict() for k, v in self.children.items()
            }
        return data


# Python type mapping for FieldType validation
_FIELD_TYPE_MAP: dict[FieldType, tuple[type, ...]] = {
    FieldType.STRING: (str,),
    FieldType.INTEGER: (int,),
    FieldType.FLOAT: (int, float),
    FieldType.BOOLEAN: (bool,),
    FieldType.DICT: (dict,),
    FieldType.LIST: (list,),
}


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------


def _build_model_schema() -> SchemaField:
    """Build the schema for a single model configuration."""
    return SchemaField(
        name="model",
        field_type=FieldType.OBJECT,
        description="Configuration for a specific model.",
        category=FieldCategory.MODEL,
        children={
            "name": SchemaField(
                name="name",
                field_type=FieldType.STRING,
                description="The model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514').",
                constraint=FieldConstraint(required=True, min_length=1),
                category=FieldCategory.MODEL,
                examples=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
            ),
            "display_name": SchemaField(
                name="display_name",
                field_type=FieldType.STRING,
                description="Human-readable name for the model.",
                default="",
                category=FieldCategory.MODEL,
                examples=["GPT-4o", "Claude Sonnet 4"],
            ),
            "max_tokens": SchemaField(
                name="max_tokens",
                field_type=FieldType.INTEGER,
                description="Maximum token limit for this model.",
                default=4096,
                category=FieldCategory.MODEL,
                constraint=FieldConstraint(min_value=1, max_value=2_000_000),
                examples=[4096, 8192, 16384, 100000],
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
            "extra": SchemaField(
                name="extra",
                field_type=FieldType.DICT,
                description="Additional model-specific configuration.",
                default={},
                category=FieldCategory.MODEL,
            ),
        },
    )


def _build_provider_schema() -> SchemaField:
    """Build the schema for a single provider configuration."""
    return SchemaField(
        name="provider",
        field_type=FieldType.OBJECT,
        description="Configuration for a model provider.",
        category=FieldCategory.PROVIDER,
        children={
            "name": SchemaField(
                name="name",
                field_type=FieldType.STRING,
                description="Unique identifier for this provider.",
                constraint=FieldConstraint(required=True, min_length=1),
                category=FieldCategory.PROVIDER,
                examples=["openai", "anthropic", "azure", "google"],
            ),
            "display_name": SchemaField(
                name="display_name",
                field_type=FieldType.STRING,
                description="Human-readable provider name.",
                default="",
                category=FieldCategory.PROVIDER,
            ),
            "api_base": SchemaField(
                name="api_base",
                field_type=FieldType.STRING,
                description="Base URL for the provider's API.",
                constraint=FieldConstraint(
                    required=True,
                    pattern=r"^https?://\S+$",
                ),
                category=FieldCategory.PROVIDER,
                examples=[
                    "https://api.openai.com/v1",
                    "https://api.anthropic.com/v1",
                ],
            ),
            "api_key_env_var": SchemaField(
                name="api_key_env_var",
                field_type=FieldType.STRING,
                description="Environment variable name containing the API key.",
                default="",
                category=FieldCategory.PROVIDER,
                examples=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
            ),
            "auth_type": SchemaField(
                name="auth_type",
                field_type=FieldType.ENUM,
                description="Authentication method used by this provider.",
                default="api_key",
                constraint=FieldConstraint(
                    allowed_values=["api_key", "bearer_token", "none"],
                ),
                category=FieldCategory.PROVIDER,
            ),
            "default_model": SchemaField(
                name="default_model",
                field_type=FieldType.STRING,
                description="Default model to use for this provider.",
                default="",
                category=FieldCategory.PROVIDER,
            ),
            "models": SchemaField(
                name="models",
                field_type=FieldType.DICT,
                description="Dictionary of available models keyed by model name.",
                default={},
                category=FieldCategory.PROVIDER,
                children={"*": _build_model_schema()},
            ),
            "headers": SchemaField(
                name="headers",
                field_type=FieldType.DICT,
                description="Additional HTTP headers to include in requests.",
                default={},
                category=FieldCategory.PROVIDER,
            ),
            "extra": SchemaField(
                name="extra",
                field_type=FieldType.DICT,
                description="Additional provider-specific configuration.",
                default={},
                category=FieldCategory.PROVIDER,
            ),
            "enabled": SchemaField(
                name="enabled",
                field_type=FieldType.BOOLEAN,
                description="Whether this provider is currently enabled.",
                default=True,
                category=FieldCategory.PROVIDER,
            ),
        },
    )


def _build_gateway_schema() -> SchemaField:
    """Build the complete gateway configuration schema."""
    return SchemaField(
        name="gateway",
        field_type=FieldType.OBJECT,
        description="Top-level gateway configuration.",
        category=FieldCategory.GATEWAY,
        children={
            "default_provider": SchemaField(
                name="default_provider",
                field_type=FieldType.STRING,
                description="Name of the default provider to use.",
                default="",
                category=FieldCategory.GATEWAY,
                env_var="GATEWAY_DEFAULT_PROVIDER",
                examples=["openai", "anthropic"],
            ),
            "providers": SchemaField(
                name="providers",
                field_type=FieldType.DICT,
                description="Dictionary of configured providers keyed by name.",
                default={},
                category=FieldCategory.PROVIDER,
                children={"*": _build_provider_schema()},
            ),
            "log_level": SchemaField(
                name="log_level",
                field_type=FieldType.ENUM,
                description="Logging level.",
                default="info",
                constraint=FieldConstraint(
                    allowed_values=[
                        "debug",
                        "info",
                        "warning",
                        "error",
                        "critical",
                    ],
                ),
                category=FieldCategory.LOGGING,
                env_var="GATEWAY_LOG_LEVEL",
            ),
            "timeout": SchemaField(
                name="timeout",
                field_type=FieldType.INTEGER,
                description="Default request timeout in seconds.",
                default=30,
                constraint=FieldConstraint(min_value=1, max_value=600),
                category=FieldCategory.GATEWAY,
                env_var="GATEWAY_TIMEOUT",
                examples=[30, 60, 120],
            ),
            "max_retries": SchemaField(
                name="max_retries",
                field_type=FieldType.INTEGER,
                description="Default number of retries for failed requests.",
                default=3,
                constraint=FieldConstraint(min_value=0, max_value=20),
                category=FieldCategory.RETRY,
                env_var="GATEWAY_MAX_RETRIES",
                examples=[0, 3, 5],
            ),
            "extra": SchemaField(
                name="extra",
                field_type=FieldType.DICT,
                description="Additional gateway-wide configuration.",
                default={},
                category=FieldCategory.GATEWAY,
            ),
        },
    )


# The canonical gateway configuration schema instance.
GATEWAY_SCHEMA: SchemaField = _build_gateway_schema()

# Convenience: provider and model sub-schemas.
PROVIDER_SCHEMA: SchemaField = _build_provider_schema()
MODEL_SCHEMA: SchemaField = _build_model_schema()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_field(path: str) -> Optional[SchemaField]:
    """Look up a schema field by dotted path.

    Args:
        path: Dotted path such as 'timeout', 'providers', or
              'providers.*.models.*.max_tokens'.

    Returns:
        The matching SchemaField, or None if the path is invalid.
    """
    parts = path.split(".")
    current = GATEWAY_SCHEMA
    for part in parts:
        child = current.get_child(part)
        if child is None:
            # Try wildcard
            child = current.get_child("*")
        if child is None:
            return None
        current = child
    return current


def list_fields(
    category: Optional[FieldCategory] = None,
    include_nested: bool = True,
) -> list[SchemaField]:
    """List all schema fields, optionally filtered by category.

    Args:
        category: Filter to only fields in this category.
        include_nested: Whether to include nested child fields.

    Returns:
        List of matching SchemaField instances.
    """
    result: list[SchemaField] = []
    _collect_fields(GATEWAY_SCHEMA, category, include_nested, result)
    return result


def _collect_fields(
    schema: SchemaField,
    category: Optional[FieldCategory],
    include_nested: bool,
    result: list[SchemaField],
) -> None:
    """Recursively collect fields from a schema tree."""
    if category is None or schema.category == category:
        result.append(schema)
    if include_nested:
        for child in schema.children.values():
            _collect_fields(child, category, include_nested, result)


def get_defaults() -> dict[str, Any]:
    """Extract default values from the schema into a flat dictionary.

    Returns:
        Dictionary of field name to default value for all fields
        that have defaults defined.
    """
    defaults: dict[str, Any] = {}
    _collect_defaults(GATEWAY_SCHEMA, "", defaults)
    return defaults


def _collect_defaults(
    schema: SchemaField, prefix: str, defaults: dict[str, Any]
) -> None:
    """Recursively collect default values."""
    path = f"{prefix}.{schema.name}" if prefix else schema.name
    if schema.default is not None:
        defaults[path] = schema.default
    for child in schema.children.values():
        if child.name != "*":
            _collect_defaults(child, path, defaults)


def generate_documented_config() -> str:
    """Generate a documented YAML configuration template.

    Returns:
        A multi-line YAML string with comments explaining each field.
    """
    lines: list[str] = [
        "# Claude Code Model Gateway Configuration",
        "# Generated from schema definitions",
        "#",
        "# See documentation for full details on each field.",
        "",
    ]
    _generate_documented_yaml(GATEWAY_SCHEMA, lines, indent=0, skip_root=True)
    return "\n".join(lines) + "\n"


def _generate_documented_yaml(
    schema: SchemaField,
    lines: list[str],
    indent: int,
    skip_root: bool = False,
) -> None:
    """Recursively generate documented YAML lines."""
    prefix = "  " * indent
    if not skip_root:
        # Comment with description
        lines.append(f"{prefix}# {schema.description}")
        if schema.constraint.allowed_values:
            lines.append(
                f"{prefix}# Allowed values: "
                f"{', '.join(schema.constraint.allowed_values)}"
            )
        if schema.env_var:
            lines.append(f"{prefix}# Env override: {schema.env_var}")
        if schema.examples:
            examples_str = ", ".join(str(e) for e in schema.examples[:3])
            lines.append(f"{prefix}# Examples: {examples_str}")

        # The field itself
        if schema.default is not None and schema.field_type not in (
            FieldType.DICT,
            FieldType.OBJECT,
        ):
            default_val = schema.default
            if isinstance(default_val, bool):
                default_val = str(default_val).lower()
            elif isinstance(default_val, str) and default_val:
                default_val = f'"{default_val}"'
            elif isinstance(default_val, str) and not default_val:
                default_val = '""'
            lines.append(f"{prefix}{schema.name}: {default_val}")
        elif schema.field_type in (FieldType.DICT, FieldType.OBJECT):
            if schema.children and not all(
                c.name == "*" for c in schema.children.values()
            ):
                lines.append(f"{prefix}{schema.name}:")
            else:
                lines.append(f"{prefix}{schema.name}: {{}}")
        else:
            lines.append(f"{prefix}{schema.name}: null")
        lines.append("")

    # Process non-wildcard children
    for child in schema.children.values():
        if child.name != "*":
            _generate_documented_yaml(child, lines, indent if skip_root else indent + 1)


def validate_against_schema(
    data: dict[str, Any],
    schema: Optional[SchemaField] = None,
) -> list[str]:
    """Validate a raw config dictionary against the schema.

    Performs type-level validation and constraint checking based on
    the schema definitions.  This is a lighter-weight check than the
    full ``ConfigValidator`` -- suitable for quick pre-flight checks.

    Args:
        data: Raw configuration dictionary.
        schema: Schema to validate against.  Defaults to GATEWAY_SCHEMA.

    Returns:
        List of validation error messages (empty if valid).
    """
    if schema is None:
        schema = GATEWAY_SCHEMA
    errors: list[str] = []
    _validate_dict_against_schema(data, schema, "", errors)
    return errors


def _validate_dict_against_schema(
    data: dict[str, Any],
    schema: SchemaField,
    path: str,
    errors: list[str],
) -> None:
    """Recursively validate a dict against a schema field."""
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        child = schema.get_child(key)
        if child is None:
            child = schema.get_child("*")
        if child is None:
            # Unknown field -- not necessarily an error for extensible configs
            continue

        # Validate the value
        field_errors = child.validate_value(value)
        for err in field_errors:
            errors.append(f"{current_path}: {err}")

        # Recurse into dicts
        if isinstance(value, dict) and child.children:
            _validate_dict_against_schema(value, child, current_path, errors)

    # Check required fields
    for child_name, child_schema in schema.children.items():
        if child_name == "*":
            continue
        if child_schema.constraint.required and child_name not in data:
            child_path = f"{path}.{child_name}" if path else child_name
            errors.append(f"{child_path}: Required field is missing.")
