"""Shared text normalization used by the deterministic graders and the
generator's dedup pass."""


def normalize(text: str) -> str:
    """Casefold + collapse whitespace + strip surrounding punctuation.

    So 'ICA.', ' ica ', and 'ICA' all compare equal without an LLM call.
    """
    cleaned = " ".join((text or "").lower().split())
    return cleaned.strip(" \t\n.,;:!?\"'`()[]")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two short strings (iterative DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def is_typo_match(user: str, accepted: str) -> bool:
    """True if ``user`` equals ``accepted`` modulo a small typo.

    Both are normalized first. An exact match always counts. Otherwise we allow a
    bounded edit distance — but only for accepted answers long enough that a near
    miss is almost certainly a misspelling rather than a different word: 1 edit for
    5-7 char answers, 2 for longer. Short answers (codes, acronyms like 'PA', 'ICA')
    must match exactly, since one character there changes the meaning.
    """
    u, c = normalize(user), normalize(accepted)
    if not u or not c:
        return False
    if u == c:
        return True
    if len(c) < 5:
        return False
    max_dist = 1 if len(c) <= 7 else 2
    if abs(len(u) - len(c)) > max_dist:
        return False
    return edit_distance(u, c) <= max_dist
