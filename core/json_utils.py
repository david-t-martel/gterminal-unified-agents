#!/usr/bin/env python3
"""
Standalone JSON utilities for testing - no external dependencies.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class JSONExtractionError(Exception):
    """Exception raised when JSON extraction fails."""

    pass


class RobustJSONExtractor:
    """Robust JSON extractor with multiple fallback strategies."""

    def __init__(self):
        # Patterns for JSON extraction
        self.patterns = [
            # JSON in markdown code blocks
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"```(?:json)?\s*(\[.*?\])\s*```",
            # JSON without markdown
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
            r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])",
        ]

    def extract_json(self, text: str, expected_type: str = "auto") -> dict[str, Any]:
        """Extract and parse JSON from text with multiple fallback strategies."""
        if not text or not isinstance(text, str):
            raise JSONExtractionError("Empty or invalid input text")

        # Strategy 1: Try direct parsing first
        try:
            result = json.loads(text.strip())
            return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Try pattern-based extraction
        for pattern in self.patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)

            for match in matches:
                try:
                    # Try parsing the match directly
                    result = json.loads(match.strip())
                    return result
                except (json.JSONDecodeError, ValueError):
                    # Try repair strategies
                    repaired = self._repair_json(match.strip())
                    if repaired:
                        try:
                            result = json.loads(repaired)
                            return result
                        except (json.JSONDecodeError, ValueError):
                            continue

        # Strategy 3: Try finding object boundaries manually
        json_candidates = self._find_json_boundaries(text)
        for candidate in json_candidates:
            try:
                result = json.loads(candidate)
                return result
            except (json.JSONDecodeError, ValueError):
                continue

        raise JSONExtractionError(f"Failed to extract valid JSON from text: {text[:200]}...")

    def _repair_json(self, json_str: str) -> str | None:
        """Try to repair malformed JSON using various strategies."""
        repaired = json_str

        # Fix trailing commas
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)

        # Convert single quotes to double quotes
        repaired = repaired.replace("'", '"')

        # Fix Python booleans
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)

        try:
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, ValueError):
            return None

    def _find_json_boundaries(self, text: str) -> list[str]:
        """Find JSON object/array boundaries in text."""
        candidates = []

        # Find potential start positions
        for start_char, end_char in [("{", " }"), ("[", "]")]:
            start_positions = [i for i, char in enumerate(text) if char == start_char]

            for start_pos in start_positions:
                # Find matching closing bracket
                bracket_count = 0
                for i, char in enumerate(text[start_pos:], start_pos):
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            # Found complete JSON object/array
                            candidate = text[start_pos : i + 1]
                            if len(candidate) > 2:  # Ignore empty objects/arrays
                                candidates.append(candidate)
                            break

        return candidates


class ParameterValidator:
    """Simple parameter validator."""

    def validate_parameters(
        self, tool_name: str, params: dict[str, Any], schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Basic parameter validation and sanitization."""
        if not isinstance(params, dict):
            raise ValueError(f"Parameters must be a dictionary, got {type(params)}")

        # If no schema, just sanitize
        if not schema:
            return self._basic_sanitization(params)

        validated = {}

        # Check required parameters
        required = schema.get("required", [])
        for param_name in required:
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")

        # Process each parameter
        param_schemas = schema.get("parameters", {})
        for param_name, param_value in params.items():
            if param_name in param_schemas:
                param_schema = param_schemas[param_name]
                validated[param_name] = self._convert_type(
                    param_value, param_schema.get("type", "str")
                )
            else:
                validated[param_name] = param_value

        # Add defaults
        for param_name, param_schema in param_schemas.items():
            if param_name not in validated and "default" in param_schema:
                validated[param_name] = param_schema["default"]

        return validated

    def _basic_sanitization(self, params: dict[str, Any]) -> dict[str, Any]:
        """Basic parameter sanitization."""
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str):
                value = value.strip()[:10000]  # Limit length
            sanitized[key] = value
        return sanitized

    def _convert_type(self, value: Any, expected_type: str) -> Any:
        """Simple type conversion."""
        if expected_type == "str":
            return str(value) if value is not None else ""
        elif expected_type == "int":
            return int(float(value)) if isinstance(value, str) else int(value)
        elif expected_type == "float":
            return float(value)
        elif expected_type == "bool":
            if isinstance(value, str):
                return value.lower() in ["true", "1", "yes", "on"]
            return bool(value)
        elif expected_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            return [value]
        else:
            return value


class SchemaValidator:
    """Simple schema validator."""

    def validate_plan(self, data: dict[str, Any]) -> bool:
        """Validate plan data."""
        return isinstance(data, dict) and "goal" in data and "steps" in data

    def validate_analysis(self, data: dict[str, Any]) -> bool:
        """Validate analysis data."""
        return isinstance(data, dict) and "complexity" in data and "estimated_steps" in data


if __name__ == "__main__":
    # Quick test
    extractor = RobustJSONExtractor()

    test_cases = [
        '```json\n{"goal": "test", "steps": []}\n```',
        'Here is the plan: {"goal": "test", "steps": []}',
        '{"goal": "test", "steps": [],}',  # trailing comma
        "{'goal': 'test', 'enabled': True}",  # single quotes + Python bool
    ]

    print("Testing JSON extraction:")
    for i, test in enumerate(test_cases):
        try:
            result = extractor.extract_json(test)
            print(f"Test {i + 1}: ✅ {result}")
        except Exception as e:
            print(f"Test {i + 1}: ❌ {e}")

    # Test parameter validator
    validator = ParameterValidator()
    schema = {
        "required": ["message"],
        "parameters": {
            "message": {"type": "str"},
            "count": {"type": "int", "default": 1},
        },
    }

    print("\nTesting parameter validation:")
    test_params = {"message": "hello", "count": "5"}
    try:
        result = validator.validate_parameters("test", test_params, schema)
        print(f"✅ Validated: {result}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
