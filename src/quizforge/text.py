"""Shared text normalization used by the deterministic graders and the
generator's dedup pass."""


def normalize(text: str) -> str:
    """Casefold + collapse whitespace + strip surrounding punctuation.

    So 'ICA.', ' ica ', and 'ICA' all compare equal without an LLM call.
    """
    cleaned = " ".join((text or "").lower().split())
    return cleaned.strip(" \t\n.,;:!?\"'`()[]")
