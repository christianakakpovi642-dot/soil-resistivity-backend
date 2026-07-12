"""Comparaison simple de versions sémantiques (ex: "1.2.0" vs "1.10.3")."""


def _parse(version: str) -> tuple[int, ...]:
    parts = []
    for segment in version.strip().split("."):
        digits = "".join(c for c in segment if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def compare_versions(v1: str, v2: str) -> int:
    """Retourne -1 si v1 < v2, 0 si égales, 1 si v1 > v2."""
    p1, p2 = _parse(v1), _parse(v2)
    length = max(len(p1), len(p2))
    p1 = p1 + (0,) * (length - len(p1))
    p2 = p2 + (0,) * (length - len(p2))

    if p1 < p2:
        return -1
    if p1 > p2:
        return 1
    return 0


def is_older_than(version: str, reference: str) -> bool:
    return compare_versions(version, reference) < 0
