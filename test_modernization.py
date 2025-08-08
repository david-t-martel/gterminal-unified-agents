#!/usr/bin/env python3
"""Test file for Python 3.12+ modernization with Pydantic V2 patterns."""

import asyncio

from pydantic import BaseModel
from pydantic import field_validator


class ModernModel(BaseModel):
    """Modern Pydantic V2 style model with Python 3.12+ syntax."""

    # Python 3.12+ union syntax with proper typing
    name: str | None = None
    age: int | None = None
    tags: list[str] = []
    metadata: dict[str, str | int] = {}

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("Age must be positive")
        return v

    # Pydantic V2 configuration using model_config
    model_config = {
        "populate_by_name": True,
        "validate_assignment": True,
    }


async def old_async_pattern():
    """Old async pattern that could be improved."""
    try:
        result = await some_async_operation()
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def old_conditionals(value: str | int | None):
    """Old conditional patterns that could use match/case."""
    if isinstance(value, str):
        return f"String: {value}"
    elif isinstance(value, int):
        return f"Integer: {value}"
    elif value is None:
        return "None value"
    else:
        return "Unknown type"


async def some_async_operation():
    """Dummy async operation."""
    await asyncio.sleep(0.1)
    return "result"


if __name__ == "__main__":
    # Test the old patterns
    model = OldStyleModel(name="test", age=25)
    print(f"Model: {model}")

    # Test async
    result = asyncio.run(old_async_pattern())
    print(f"Async result: {result}")

    # Test conditionals
    print(old_conditionals("hello"))
    print(old_conditionals(42))
    print(old_conditionals(None))
