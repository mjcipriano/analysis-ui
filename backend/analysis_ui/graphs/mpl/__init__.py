"""matplotlib engine (placeholder — replaced by the real implementation)."""
from ...core.errors import AnalysisError


def render(payload: dict) -> bytes:
    raise AnalysisError(f"The matplotlib engine cannot draw “{payload['type']}” yet.")
