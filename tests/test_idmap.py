import asyncio

from backend.core.idmap import resolve, resolve_local

KNOWN = {p.lower(): p for p in ["3001", "26603", "41879a", "3626b", "Mixed1"]}


def test_exact_match_returns_real_casing():
    assert resolve_local("3001", KNOWN) == "3001"
    assert resolve_local("mixed1", KNOWN) == "Mixed1"


def test_strip_trailing_letter_variant():
    assert resolve_local("3001a", KNOWN) == "3001"


def test_strip_print_suffix():
    assert resolve_local("26603pb509", KNOWN) == "26603"


def test_append_letter_when_rebrickable_has_one():
    assert resolve_local("41879", KNOWN) == "41879a"
    assert resolve_local("41879pb014", KNOWN) == "41879a"


def test_unknown_is_none_locally():
    assert resolve_local("970c00pb1023", KNOWN) is None


def test_api_fallback_used_only_when_local_fails():
    calls = []

    async def api(bl):
        calls.append(bl)
        return {"970c00": "73200"}.get(bl)

    assert asyncio.run(resolve("3001a", KNOWN, api)) == ("3001", False)
    assert calls == []
    assert asyncio.run(resolve("970c00", KNOWN, api)) == ("73200", False)
    assert calls == ["970c00"]


def test_unmapped_is_flagged_and_api_errors_swallowed():
    async def boom(bl):
        raise RuntimeError("down")

    assert asyncio.run(resolve("zzz", KNOWN, boom)) == ("zzz", True)
    assert asyncio.run(resolve("zzz", KNOWN, None)) == ("zzz", True)
