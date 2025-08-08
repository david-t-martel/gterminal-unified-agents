class BaseModel:
    """Base model class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> None:
        return f"BaseModel(name='{self.name}')"
