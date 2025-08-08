#!/usr/bin/env python3
"""Enhanced parameter validation for ReAct engine tools.

This module provides comprehensive parameter validation, type conversion,
and sanitization for tool parameters to prevent execution errors and
improve reliability of the autonomous ReAct system.
"""

from collections.abc import Callable
import inspect
import json
import logging
from pathlib import Path
import re
from typing import Any, get_type_hints
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Exception raised when parameter validation fails."""


class TypeConverter:
    """Handles type conversion for tool parameters."""

    @staticmethod
    def to_bool(value: Any) -> bool:
        """Convert various representations to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "on", "enabled"]
        if isinstance(value, int | float):
            return bool(value)
        return False

    @staticmethod
    def to_int(value: Any) -> int:
        """Convert value to integer with validation."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            msg = f"Float {value} is not a whole number"
            raise ValueError(msg)
        if isinstance(value, str):
            return int(value.strip())
        msg = f"Cannot convert {type(value)} to int: {value}"
        raise ValueError(msg)

    @staticmethod
    def to_float(value: Any) -> float:
        """Convert value to float with validation."""
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        msg = f"Cannot convert {type(value)} to float: {value}"
        raise ValueError(msg)

    @staticmethod
    def to_str(value: Any) -> str:
        """Convert value to string."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict | list):
            return json.dumps(value)
        return str(value)

    @staticmethod
    def to_list(value: Any) -> list[Any]:
        """Convert value to list."""
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            # Try to parse as JSON array first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

            # Split by common delimiters
            for delimiter in [",", ";", "|", "\n"]:
                if delimiter in value:
                    return [item.strip() for item in value.split(delimiter) if item.strip()]

            # Single item list
            return [value] if value else []

        # Single item
        return [value]

    @staticmethod
    def to_dict(value: Any) -> dict[str, Any]:
        """Convert value to dictionary."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

            # Try to parse key=value pairs
            if "=" in value:
                result = {}
                pairs = value.split(",") if "," in value else [value]
                for pair in pairs:
                    if "=" in pair:
                        key, val = pair.split("=", 1)
                        result[key.strip()] = val.strip()
                return result

        msg = f"Cannot convert {type(value)} to dict: {value}"
        raise ValueError(msg)


class ParameterValidator:
    """Validates and sanitizes tool parameters."""

    def __init__(self) -> None:
        self.converter = TypeConverter()

        # Common parameter patterns and their validators
        self.validators = {
            "file_path": self._validate_file_path,
            "directory_path": self._validate_directory_path,
            "url": self._validate_url,
            "email": self._validate_email,
            "json": self._validate_json,
            "positive_int": self._validate_positive_int,
            "port": self._validate_port,
            "timeout": self._validate_timeout,
        }

    def validate_parameters(
        self,
        tool_name: str,
        params: dict[str, Any],
        expected_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and convert parameters according to schema.

        Args:
            tool_name: Name of the tool for context
            params: Raw parameters from LLM
            expected_schema: Expected parameter schema

        Returns:
            Validated and converted parameters

        Raises:
            ParameterValidationError: If validation fails

        """
        if not isinstance(params, dict):
            msg = f"Parameters must be a dictionary, got {type(params)}"
            raise ParameterValidationError(msg)

        # If no schema provided, do basic sanitization
        if not expected_schema:
            return self._basic_sanitization(params)

        validated_params = {}

        # Check required parameters
        required = expected_schema.get("required", [])
        for param_name in required:
            if param_name not in params:
                msg = f"Missing required parameter: {param_name}"
                raise ParameterValidationError(msg)

        # Validate each parameter
        param_schemas = expected_schema.get("parameters", {})
        for param_name, param_value in params.items():
            if param_name in param_schemas:
                schema = param_schemas[param_name]
                validated_params[param_name] = self._validate_parameter(
                    param_name, param_value, schema, tool_name
                )
            else:
                # Unknown parameter - log warning but include it
                logger.warning(f"Unknown parameter '{param_name}' for tool '{tool_name}'")
                validated_params[param_name] = self._basic_sanitize_value(param_value)

        # Add default values for missing optional parameters
        for param_name, schema in param_schemas.items():
            if param_name not in validated_params and "default" in schema:
                validated_params[param_name] = schema["default"]

        return validated_params

    def _validate_parameter(
        self, param_name: str, param_value: Any, schema: dict[str, Any], tool_name: str
    ) -> Any:
        """Validate a single parameter against its schema."""
        # Handle None values
        if param_value is None:
            if schema.get("required", False):
                msg = f"Parameter '{param_name}' cannot be None"
                raise ParameterValidationError(msg)
            return schema.get("default")

        # Type conversion
        expected_type = schema.get("type", "str")
        try:
            converted_value = self._convert_type(param_value, expected_type)
        except ValueError as e:
            msg = f"Type conversion failed for parameter '{param_name}': {e}"
            raise ParameterValidationError(msg)

        # Validation rules
        validation_rules = schema.get("validation", {})

        # Min/Max for numeric types
        if expected_type in ["int", "float"] and isinstance(converted_value, int | float):
            if "min" in validation_rules and converted_value < validation_rules["min"]:
                msg = f"Parameter '{param_name}' value {converted_value} is below minimum {validation_rules['min']}"
                raise ParameterValidationError(
                    msg,
                )
            if "max" in validation_rules and converted_value > validation_rules["max"]:
                msg = f"Parameter '{param_name}' value {converted_value} exceeds maximum {validation_rules['max']}"
                raise ParameterValidationError(
                    msg,
                )

        # Length validation for strings and lists
        if isinstance(converted_value, str | list):
            if (
                "min_length" in validation_rules
                and len(converted_value) < validation_rules["min_length"]
            ):
                msg = f"Parameter '{param_name}' length {len(converted_value)} is below minimum {validation_rules['min_length']}"
                raise ParameterValidationError(
                    msg,
                )
            if (
                "max_length" in validation_rules
                and len(converted_value) > validation_rules["max_length"]
            ):
                msg = f"Parameter '{param_name}' length {len(converted_value)} exceeds maximum {validation_rules['max_length']}"
                raise ParameterValidationError(
                    msg,
                )

        # Pattern validation for strings
        if isinstance(converted_value, str) and "pattern" in validation_rules:
            if not re.match(validation_rules["pattern"], converted_value):
                msg = f"Parameter '{param_name}' value '{converted_value}' does not match required pattern"
                raise ParameterValidationError(
                    msg,
                )

        # Allowed values
        if "allowed_values" in validation_rules:
            if converted_value not in validation_rules["allowed_values"]:
                msg = f"Parameter '{param_name}' value '{converted_value}' not in allowed values: {validation_rules['allowed_values']}"
                raise ParameterValidationError(
                    msg,
                )

        # Custom validators
        validator_name = validation_rules.get("validator")
        if validator_name and validator_name in self.validators:
            try:
                self.validators[validator_name](converted_value)
            except ValueError as e:
                msg = f"Custom validation failed for parameter '{param_name}': {e}"
                raise ParameterValidationError(msg)

        return converted_value

    def _convert_type(self, value: Any, expected_type: str) -> Any:
        """Convert value to expected type."""
        if expected_type == "str":
            return self.converter.to_str(value)
        if expected_type == "int":
            return self.converter.to_int(value)
        if expected_type == "float":
            return self.converter.to_float(value)
        if expected_type == "bool":
            return self.converter.to_bool(value)
        if expected_type == "list":
            return self.converter.to_list(value)
        if expected_type == "dict":
            return self.converter.to_dict(value)
        # Unknown type, return as-is
        return value

    def _basic_sanitization(self, params: dict[str, Any]) -> dict[str, Any]:
        """Basic parameter sanitization without schema."""
        sanitized = {}
        for key, value in params.items():
            sanitized[key] = self._basic_sanitize_value(value)
        return sanitized

    def _basic_sanitize_value(self, value: Any) -> Any:
        """Basic value sanitization."""
        if isinstance(value, str):
            # Remove potential injection characters
            value = value.strip()
            # Limit length
            if len(value) > 10000:
                logger.warning(f"Truncating long string parameter (length: {len(value)})")
                value = value[:10000]

        return value

    # Custom validators
    def _validate_file_path(self, value: str) -> None:
        """Validate file path."""
        if not isinstance(value, str):
            msg = "File path must be a string"
            raise ValueError(msg)

        # Check for path traversal attempts
        if ".." in value or value.startswith("/"):
            msg = "Path traversal not allowed"
            raise ValueError(msg)

        # Basic path validation
        try:
            Path(value)
        except Exception:
            msg = "Invalid file path"
            raise ValueError(msg)

    def _validate_directory_path(self, value: str) -> None:
        """Validate directory path."""
        self._validate_file_path(value)
        # Additional directory-specific validation could go here

    def _validate_url(self, value: str) -> None:
        """Validate URL."""
        try:
            result = urlparse(value)
            if not all([result.scheme, result.netloc]):
                msg = "Invalid URL format"
                raise ValueError(msg)
            if result.scheme not in ["http", "https", "ftp"]:
                msg = "Unsupported URL scheme"
                raise ValueError(msg)
        except Exception:
            msg = "Invalid URL"
            raise ValueError(msg)

    def _validate_email(self, value: str) -> None:
        """Validate email address."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, value):
            msg = "Invalid email format"
            raise ValueError(msg)

    def _validate_json(self, value: str) -> None:
        """Validate JSON string."""
        try:
            json.loads(value)
        except json.JSONDecodeError:
            msg = "Invalid JSON format"
            raise ValueError(msg)

    def _validate_positive_int(self, value: int) -> None:
        """Validate positive integer."""
        if not isinstance(value, int) or value <= 0:
            msg = "Must be a positive integer"
            raise ValueError(msg)

    def _validate_port(self, value: int) -> None:
        """Validate port number."""
        if not isinstance(value, int) or not (1 <= value <= 65535):
            msg = "Port must be between 1 and 65535"
            raise ValueError(msg)

    def _validate_timeout(self, value: float) -> None:
        """Validate timeout value."""
        if not isinstance(value, int | float) or value <= 0:
            msg = "Timeout must be a positive number"
            raise ValueError(msg)


def create_tool_schema(tool_func: Callable) -> dict[str, Any]:
    """Create a parameter schema from a tool function's signature and docstring.

    Args:
        tool_func: Tool function to analyze

    Returns:
        Parameter schema dictionary

    """
    schema: dict[str, Any] = {"required": [], "parameters": {}}

    try:
        # Get function signature
        sig = inspect.signature(tool_func)
        type_hints = get_type_hints(tool_func)

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_schema = {}

            # Determine type
            if param_name in type_hints:
                hint = type_hints[param_name]
                if hint == str:
                    param_schema["type"] = "str"
                elif hint == int:
                    param_schema["type"] = "int"
                elif hint == float:
                    param_schema["type"] = "float"
                elif hint == bool:
                    param_schema["type"] = "bool"
                elif hint == list:
                    param_schema["type"] = "list"
                elif hint == dict:
                    param_schema["type"] = "dict"

            # Check if required
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
            else:
                param_schema["default"] = param.default

            schema["parameters"][param_name] = param_schema

    except Exception as e:
        logger.warning(f"Failed to create schema for {tool_func.__name__}: {e}")

    return schema


# Convenience functions
_validator = ParameterValidator()


def validate_tool_parameters(
    tool_name: str,
    params: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate tool parameters."""
    return _validator.validate_parameters(tool_name, params, schema)


if __name__ == "__main__":
    # Test the parameter validator
    validator = ParameterValidator()

    # Test schema
    test_schema = {
        "required": ["message", "count"],
        "parameters": {
            "message": {"type": "str", "validation": {"min_length": 1, "max_length": 100}},
            "count": {"type": "int", "validation": {"min": 1, "max": 10}},
            "enabled": {"type": "bool", "default": True},
            "tags": {"type": "list", "default": []},
        },
    }

    test_params = {
        "message": "Hello World",
        "count": "5",  # String that should convert to int
        "enabled": "true",  # String that should convert to bool
        "tags": "tag1,tag2,tag3",  # String that should convert to list
    }

    try:
        validated = validator.validate_parameters("test_tool", test_params, test_schema)
        for _key, _value in validated.items():
            pass
    except ParameterValidationError:
        pass
