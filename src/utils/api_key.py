import os


class ApiKeyRequiredError(Exception):
    def __init__(self, feature: str):
        self.feature = feature
        super().__init__(
            f"{feature} requires OPENAI_API_KEY or OPENROUTER_API_KEY. "
            "Set one in .env or the environment and try again."
        )


def get_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def require_api_key_for_query() -> None:
    if not get_api_key():
        raise ApiKeyRequiredError("Query (Q&A)")


def require_api_key_for_vision() -> None:
    if not get_api_key():
        raise ApiKeyRequiredError("Vision-based extraction (scanned PDFs)")
