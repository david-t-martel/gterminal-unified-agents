class Config:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def __repr__(self) -> None:
        return f"Config(api_key='{self.api_key}')"
