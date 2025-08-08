#!/usr/bin/env python3
"""Fully modernized Python 3.12+ example with Pydantic V2."""

import asyncio

from pydantic import BaseModel
from pydantic import field_validator


class ModernModel(BaseModel):
    """Modern Pydantic V2 model with Python 3.12+ features."""

    # Python 3.12+ union syntax
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

    # Pydantic V2 configuration
    model_config = {
        "populate_by_name": True,
        "validate_assignment": True,
    }


async def modern_async_pattern() -> str | None:
    """Modern async with specific exception handling."""
    try:
        result = await some_async_operation()
    except (ConnectionError, TimeoutError) as e:
        print(f"Connection error: {e}")
        return None
    else:
        return result


def modern_match_case(value: str | int | None) -> str:
    """Python 3.10+ match/case patterns."""
    match value:
        case str() as s:
            return f"String: {s}"
        case int() as i:
            return f"Integer: {i}"
        case None:
            return "None value"
        case _:
            return "Unknown type"


async def some_async_operation() -> str:
    """Dummy async operation."""
    await asyncio.sleep(0.1)
    return "result"


def main() -> None:
    """Test modernized patterns."""
    # Test Pydantic V2 model
    model = ModernModel(name="test", age=25)
    print(f"Model: {model}")

    # Test modern async
    result = asyncio.run(modern_async_pattern())
    print(f"Async result: {result}")

    # Test match/case
    print(modern_match_case("hello"))
    print(modern_match_case(42))
    print(modern_match_case(None))


if __name__ == "__main__":
    main()
