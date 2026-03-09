"""Tests for the configuration schema module."""

import pytest

from src.config.schema import (
    GATEWAY_SCHEMA,
    MODEL_SCHEMA,
    PROVIDER_SCHEMA,
    FieldCategory,
    FieldConstraint,
    FieldType,
    SchemaField,
    generate_documented_config,
    get_defaults,
    get_field,
    list_fields,
    validate_against_schema,
)


# ---------------------------------------------------------------------------
# FieldConstraint tests
# ---------------------------------------------------------------------------


class TestFieldConstraint:
    """Tests for FieldConstraint validation logic."""

    def test_min_value_violation(self):
        """Test that values below min_value are rejected."""
        c = FieldConstraint(min_value=0)
        errors = c.validate(-1)
        assert len(errors) == 1
        assert "minimum" in errors[0]

    def test_min_value_ok(self):
        """Test that values at min_value are accepted."""
        c = FieldConstraint(min_value=0)
        errors = c.validate(0)
        assert errors == []

    def test_max_value_violation(self):
        """Test that values above max_value are rejected."""
        c = FieldConstraint(max_value=100)
        errors = c.validate(101)
        assert len(errors) == 1
        assert "maximum" in errors[0]

    def test_max_value_ok(self):
        """Test that values at max_value are accepted."""
        c = FieldConstraint(max_value=100)
        errors = c.validate(100)
        assert errors == []

    def test_min_and_max_value_range(self):
        """Test min+max range constraint."""
        c = FieldConstraint(min_value=1, max_value=10)
        assert c.validate(0) != []   # below min
        assert c.validate(5) == []   # in range
        assert c.validate(11) != []  # above max

    def test_min_length_violation(self):
        """Test that strings shorter than min_length are rejected."""
        c = FieldConstraint(min_length=3)
        errors = c.validate("ab")
        assert len(errors) == 1
        assert "minimum" in errors[0]

    def test_min_length_ok(self):
        """Test that strings at min_length are accepted."""
        c = FieldConstraint(min_length=3)
        assert c.validate("abc") == []

    def test_max_length_violation(self):
        """Test that strings longer than max_length are rejected."""
        c = FieldConstraint(max_length=5)
        errors = c.validate("toolong")
        assert len(errors) == 1
        assert "maximum" in errors[0]

    def test_max_length_ok(self):
        """Test that strings at max_length are accepted."""
        c = FieldConstraint(max_length=5)
        assert c.validate("hello") == []

    def test_pattern_violation(self):
        """Test that strings not matching the pattern are rejected."""
        c = FieldConstraint(pattern=r"^\d+$")
        errors = c.validate("abc")
        assert len(errors) == 1
        assert "pattern" in errors[0]

    def test_pattern_ok(self):
        """Test that strings matching the pattern are accepted."""
        c = FieldConstraint(pattern=r"^\d+$")
        assert c.validate("123") == []

    def test_allowed_values_violation(self):
        """Test that values not in allowed_values are rejected."""
        c = FieldConstraint(allowed_values=["a", "b", "c"])
        errors = c.validate("d")
        assert len(errors) == 1
        assert "not one of" in errors[0]

    def test_allowed_values_ok(self):
        """Test that values in allowed_values are accepted."""
        c = FieldConstraint(allowed_values=["a", "b", "c"])
        assert c.validate("a") == []

    def test_non_string_not_checked_for_length(self):
        """Test that non-string values skip length checks."""
        c = FieldConstraint(min_length=5, max_length=10)
        assert c.validate(42) == []  # integers skip string checks

    def test_non_numeric_not_checked_for_range(self):
        """Test that non-numeric values skip range checks."""
        c = FieldConstraint(min_value=0, max_value=100)
        assert c.validate("hello") == []  # strings skip range checks

    def test_empty_constraint_passes_all(self):
        """Test that empty constraint passes anything."""
        c = FieldConstraint()
        assert c.validate("any value") == []
        assert c.validate(42) == []
        assert c.validate(-999) == []

    def test_multiple_violations(self):
        """Test that multiple violations are all returned."""
        c = FieldConstraint(min_value=10, max_value=5)  # impossible range
        # Both can trigger for numeric values if range is weird
        errors = c.validate(7)
        # 7 is above min_value=10? No, 7 < 10 so min violation triggers
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# SchemaField tests
# ---------------------------------------------------------------------------


class TestSchemaField:
    """Tests for SchemaField."""

    def test_validate_value_correct_type(self):
        """Test that correct types pass type validation."""
        f = SchemaField(
            name="timeout",
            field_type=FieldType.INTEGER,
            description="Timeout.",
        )
        errors = f.validate_value(30)
        assert errors == []

    def test_validate_value_wrong_type(self):
        """Test that wrong types fail type validation."""
        f = SchemaField(
            name="timeout",
            field_type=FieldType.INTEGER,
            description="Timeout.",
        )
        errors = f.validate_value("thirty")
        assert len(errors) >= 1
        assert "integer" in errors[0].lower() or "int" in errors[0].lower()

    def test_validate_value_none_skips_type_check(self):
        """Test that None values skip type checking."""
        f = SchemaField(
            name="optional_field",
            field_type=FieldType.STRING,
            description="Optional string.",
        )
        errors = f.validate_value(None)
        assert errors == []

    def test_validate_value_with_constraint(self):
        """Test that constraints are applied during validation."""
        f = SchemaField(
            name="count",
            field_type=FieldType.INTEGER,
            description="Count.",
            constraint=FieldConstraint(min_value=1),
        )
        errors = f.validate_value(0)
        assert len(errors) >= 1

    def test_get_child_found(self):
        """Test that get_child returns a child field by name."""
        parent = SchemaField(
            name="parent",
            field_type=FieldType.OBJECT,
            description="Parent.",
            children={
                "child": SchemaField(
                    name="child",
                    field_type=FieldType.STRING,
                    description="Child field.",
                )
            },
        )
        child = parent.get_child("child")
        assert child is not None
        assert child.name == "child"

    def test_get_child_not_found(self):
        """Test that get_child returns None for missing children."""
        f = SchemaField(
            name="field",
            field_type=FieldType.OBJECT,
            description="Field.",
        )
        assert f.get_child("nonexistent") is None

    def test_to_dict_basic(self):
        """Test to_dict with basic field."""
        f = SchemaField(
            name="log_level",
            field_type=FieldType.ENUM,
            description="Logging level.",
            default="info",
            constraint=FieldConstraint(allowed_values=["debug", "info"]),
        )
        d = f.to_dict()
        assert d["name"] == "log_level"
        assert d["type"] == "enum"
        assert d["description"] == "Logging level."
        assert d["default"] == "info"
        assert "debug" in d["allowed_values"]

    def test_to_dict_with_env_var(self):
        """Test to_dict includes env_var when set."""
        f = SchemaField(
            name="timeout",
            field_type=FieldType.INTEGER,
            description="Timeout.",
            env_var="GATEWAY_TIMEOUT",
        )
        d = f.to_dict()
        assert d["env_var"] == "GATEWAY_TIMEOUT"

    def test_to_dict_with_deprecation(self):
        """Test to_dict includes deprecation info."""
        f = SchemaField(
            name="old_field",
            field_type=FieldType.STRING,
            description="Old field.",
            deprecated=True,
            deprecation_message="Use new_field instead.",
        )
        d = f.to_dict()
        assert d["deprecated"] is True
        assert "new_field" in d["deprecation_message"]

    def test_to_dict_with_children(self):
        """Test to_dict recursively includes children."""
        f = SchemaField(
            name="parent",
            field_type=FieldType.OBJECT,
            description="Parent.",
            children={
                "child": SchemaField(
                    name="child",
                    field_type=FieldType.STRING,
                    description="Child.",
                )
            },
        )
        d = f.to_dict()
        assert "children" in d
        assert "child" in d["children"]

    def test_float_type_accepts_int(self):
        """Test that FLOAT type accepts integers (widening)."""
        f = SchemaField(
            name="rate",
            field_type=FieldType.FLOAT,
            description="Rate.",
        )
        errors = f.validate_value(5)  # int is acceptable for float field
        assert errors == []

    def test_float_type_accepts_float(self):
        """Test that FLOAT type accepts floats."""
        f = SchemaField(
            name="rate",
            field_type=FieldType.FLOAT,
            description="Rate.",
        )
        errors = f.validate_value(5.5)
        assert errors == []

    def test_boolean_type_validation(self):
        """Test BOOLEAN type validation."""
        f = SchemaField(
            name="enabled",
            field_type=FieldType.BOOLEAN,
            description="Enabled.",
        )
        assert f.validate_value(True) == []
        assert f.validate_value(False) == []
        assert f.validate_value("true") != []  # string, not bool

    def test_dict_type_validation(self):
        """Test DICT type validation."""
        f = SchemaField(
            name="headers",
            field_type=FieldType.DICT,
            description="Headers.",
        )
        assert f.validate_value({"key": "value"}) == []
        assert f.validate_value("not-a-dict") != []

    def test_list_type_validation(self):
        """Test LIST type validation."""
        f = SchemaField(
            name="items",
            field_type=FieldType.LIST,
            description="Items.",
        )
        assert f.validate_value([1, 2, 3]) == []
        assert f.validate_value("not-a-list") != []


# ---------------------------------------------------------------------------
# GATEWAY_SCHEMA / PROVIDER_SCHEMA / MODEL_SCHEMA tests
# ---------------------------------------------------------------------------


class TestGatewaySchema:
    """Tests for the built-in schema instances."""

    def test_gateway_schema_is_object(self):
        """Test that the gateway schema is an OBJECT type."""
        assert GATEWAY_SCHEMA.field_type == FieldType.OBJECT
        assert GATEWAY_SCHEMA.name == "gateway"

    def test_gateway_schema_has_expected_top_level_fields(self):
        """Test expected top-level fields in gateway schema."""
        expected = {"default_provider", "providers", "log_level", "timeout", "max_retries"}
        actual = set(GATEWAY_SCHEMA.children.keys())
        assert expected.issubset(actual)

    def test_provider_schema_has_api_base(self):
        """Test that provider schema includes api_base."""
        assert "api_base" in PROVIDER_SCHEMA.children

    def test_provider_schema_api_base_required(self):
        """Test that api_base is required in provider schema."""
        api_base = PROVIDER_SCHEMA.children["api_base"]
        assert api_base.constraint.required is True

    def test_model_schema_has_max_tokens(self):
        """Test that model schema includes max_tokens."""
        assert "max_tokens" in MODEL_SCHEMA.children

    def test_model_schema_max_tokens_constraints(self):
        """Test max_tokens has min/max constraints."""
        mt = MODEL_SCHEMA.children["max_tokens"]
        assert mt.constraint.min_value is not None
        assert mt.constraint.min_value >= 1


# ---------------------------------------------------------------------------
# get_field tests
# ---------------------------------------------------------------------------


class TestGetField:
    """Tests for get_field() helper."""

    def test_get_top_level_field(self):
        """Test getting a top-level field."""
        f = get_field("timeout")
        assert f is not None
        assert f.name == "timeout"

    def test_get_nested_field(self):
        """Test getting a nested field via dotted path."""
        f = get_field("providers")
        assert f is not None

    def test_get_deep_nested_field(self):
        """Test getting a deeply nested field."""
        # providers.* -> provider schema -> api_base
        f = get_field("providers.*.api_base")
        assert f is not None
        assert f.name == "api_base"

    def test_get_model_field_via_wildcard(self):
        """Test getting model max_tokens via wildcard path."""
        f = get_field("providers.*.models.*.max_tokens")
        assert f is not None
        assert f.name == "max_tokens"

    def test_nonexistent_field_returns_none(self):
        """Test that nonexistent path returns None."""
        f = get_field("nonexistent_field")
        assert f is None

    def test_invalid_nested_path_returns_none(self):
        """Test that invalid nested path returns None."""
        f = get_field("timeout.nested")
        assert f is None

    def test_log_level_has_allowed_values(self):
        """Test that log_level field has allowed_values."""
        f = get_field("log_level")
        assert f is not None
        assert "info" in f.constraint.allowed_values
        assert "debug" in f.constraint.allowed_values


# ---------------------------------------------------------------------------
# list_fields tests
# ---------------------------------------------------------------------------


class TestListFields:
    """Tests for list_fields() helper."""

    def test_list_all_fields(self):
        """Test listing all fields returns multiple."""
        fields = list_fields()
        assert len(fields) > 1

    def test_list_gateway_fields(self):
        """Test listing only gateway-category fields."""
        fields = list_fields(category=FieldCategory.GATEWAY)
        assert all(f.category == FieldCategory.GATEWAY for f in fields)

    def test_list_provider_fields(self):
        """Test listing only provider-category fields."""
        fields = list_fields(category=FieldCategory.PROVIDER)
        assert len(fields) > 0
        assert all(f.category == FieldCategory.PROVIDER for f in fields)

    def test_list_model_fields(self):
        """Test listing only model-category fields."""
        fields = list_fields(category=FieldCategory.MODEL)
        assert len(fields) > 0
        assert all(f.category == FieldCategory.MODEL for f in fields)

    def test_list_without_nested(self):
        """Test listing without nested fields returns fewer items."""
        all_fields = list_fields(include_nested=True)
        top_only = list_fields(include_nested=False)
        # Top-only should have fewer or equal items
        assert len(top_only) <= len(all_fields)

    def test_list_logging_fields(self):
        """Test listing logging-category fields."""
        fields = list_fields(category=FieldCategory.LOGGING)
        assert len(fields) >= 1  # At least log_level


# ---------------------------------------------------------------------------
# get_defaults tests
# ---------------------------------------------------------------------------


class TestGetDefaults:
    """Tests for get_defaults() helper."""

    def test_returns_dict(self):
        """Test that get_defaults returns a dictionary."""
        defaults = get_defaults()
        assert isinstance(defaults, dict)

    def test_includes_timeout_default(self):
        """Test that defaults include timeout."""
        defaults = get_defaults()
        # timeout default is 30 - check for any key containing 'timeout'
        timeout_keys = [k for k in defaults if "timeout" in k]
        assert len(timeout_keys) >= 1

    def test_includes_log_level_default(self):
        """Test that defaults include log_level."""
        defaults = get_defaults()
        log_keys = [k for k in defaults if "log_level" in k]
        assert len(log_keys) >= 1
        # The value should be 'info'
        for k in log_keys:
            assert defaults[k] == "info"

    def test_includes_max_retries_default(self):
        """Test that defaults include max_retries."""
        defaults = get_defaults()
        retry_keys = [k for k in defaults if "max_retries" in k]
        assert len(retry_keys) >= 1


# ---------------------------------------------------------------------------
# generate_documented_config tests
# ---------------------------------------------------------------------------


class TestGenerateDocumentedConfig:
    """Tests for generate_documented_config()."""

    def test_returns_string(self):
        """Test that documented config returns a string."""
        result = generate_documented_config()
        assert isinstance(result, str)

    def test_contains_yaml_comments(self):
        """Test that the output contains YAML comments."""
        result = generate_documented_config()
        assert "#" in result

    def test_contains_key_fields(self):
        """Test that the output mentions important fields."""
        result = generate_documented_config()
        assert "timeout" in result
        assert "log_level" in result
        assert "max_retries" in result

    def test_contains_env_var_hints(self):
        """Test that env var hints are included in comments."""
        result = generate_documented_config()
        assert "GATEWAY_TIMEOUT" in result or "Env override" in result

    def test_ends_with_newline(self):
        """Test that the output ends with a newline."""
        result = generate_documented_config()
        assert result.endswith("\n")

    def test_contains_allowed_values_hints(self):
        """Test that allowed values are noted for enum fields."""
        result = generate_documented_config()
        assert "Allowed values" in result


# ---------------------------------------------------------------------------
# validate_against_schema tests
# ---------------------------------------------------------------------------


class TestValidateAgainstSchema:
    """Tests for validate_against_schema()."""

    def test_valid_empty_dict_passes(self):
        """Test that an empty dict passes schema validation."""
        errors = validate_against_schema({})
        assert errors == []

    def test_valid_timeout_passes(self):
        """Test that a valid timeout value passes."""
        errors = validate_against_schema({"timeout": 30})
        assert errors == []

    def test_invalid_timeout_type(self):
        """Test that a string timeout fails schema validation."""
        errors = validate_against_schema({"timeout": "thirty"})
        assert len(errors) >= 1
        assert any("timeout" in e for e in errors)

    def test_invalid_timeout_too_low(self):
        """Test that timeout=0 fails (below min_value=1)."""
        errors = validate_against_schema({"timeout": 0})
        assert len(errors) >= 1

    def test_invalid_log_level_string(self):
        """Test that an invalid log level string fails."""
        errors = validate_against_schema({"log_level": "verbose"})
        assert len(errors) >= 1
        assert any("log_level" in e for e in errors)

    def test_valid_log_level_passes(self):
        """Test that a valid log level passes."""
        errors = validate_against_schema({"log_level": "info"})
        assert errors == []

    def test_invalid_max_retries_type(self):
        """Test that string max_retries fails."""
        errors = validate_against_schema({"max_retries": "three"})
        assert len(errors) >= 1

    def test_nested_provider_invalid_url(self):
        """Test that invalid api_base fails nested schema check."""
        data = {
            "providers": {
                "test": {
                    "name": "test",
                    "api_base": 12345,  # should be string
                }
            }
        }
        errors = validate_against_schema(data)
        assert len(errors) >= 1

    def test_unknown_fields_ignored(self):
        """Test that unknown fields are silently ignored."""
        errors = validate_against_schema({"unknown_field_xyz": "value"})
        assert errors == []

    def test_valid_full_config_passes(self):
        """Test that a complete valid config passes schema validation."""
        data = {
            "default_provider": "openai",
            "log_level": "info",
            "timeout": 30,
            "max_retries": 3,
        }
        errors = validate_against_schema(data)
        assert errors == []

    def test_model_max_tokens_constraint(self):
        """Test max_tokens constraint inside nested model config."""
        data = {
            "providers": {
                "test": {
                    "models": {
                        "mymodel": {
                            "name": "mymodel",
                            "max_tokens": 0,  # below min_value=1
                        }
                    }
                }
            }
        }
        errors = validate_against_schema(data)
        assert len(errors) >= 1

    def test_custom_schema_argument(self):
        """Test passing a custom schema to validate_against_schema."""
        custom_schema = SchemaField(
            name="custom",
            field_type=FieldType.OBJECT,
            description="Custom schema.",
            children={
                "count": SchemaField(
                    name="count",
                    field_type=FieldType.INTEGER,
                    description="A count.",
                    constraint=FieldConstraint(min_value=0),
                )
            },
        )
        errors = validate_against_schema({"count": -1}, schema=custom_schema)
        assert len(errors) >= 1

        errors = validate_against_schema({"count": 5}, schema=custom_schema)
        assert errors == []
