"""Shared text helpers: tokenizing, stemming, sentence spans, and concept name keys.

Sentence spans are always (start, end) offsets into the original page text, so
``text[start:end]`` is a verbatim quote. Nothing here rewrites stored text.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either else etc few for from further
had has have having he her here hers herself him himself his how however i if in into is it its itself
just may me might more most must my myself no nor not now of off on once only or other our ours
ourselves out over own per same shall she should so some such than that the their theirs them
themselves then there these they this those through thus to too under until up upon us very via was
we were what when where which while who whom why will with within without would yet you your yours
yourself yourselves
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")
_DOT_LEADER = re.compile(r"(?:\.\s?){4,}|(?:…){2,}|_{4,}")
_LEADING_DETERMINERS = ("the", "a", "an", "this", "these", "that", "those", "its", "their", "our", "such", "each")
_ABBREVIATIONS = frozenset({"e.g", "i.e", "fig", "figs", "eq", "eqs", "et al", "al", "vs", "etc", "no", "dr",
                            "mr", "ms", "prof", "inc", "ltd", "co", "vol", "pp", "p", "ch", "sec", "u.s", "approx"})


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Light English stemmer (Snowball/Porter2), cached because words repeat."""
    return _STEMMER.stemWord(word)


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords and 1-character tokens, and stem."""
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        word = match.group(0).lower().replace("’", "'")
        if word.endswith("'s"):
            word = word[:-2]
        if len(word) < 2 or word in STOPWORDS:
            continue
        tokens.append(stem(word))
    return tokens


def word_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, stem) for every non-stopword token, for highlighting."""
    spans: list[tuple[int, int, str]] = []
    for match in _TOKEN.finditer(text):
        word = match.group(0).lower()
        if len(word) < 2 or word in STOPWORDS:
            continue
        spans.append((match.start(), match.end(), stem(word)))
    return spans


def mask_dot_leaders(text: str) -> str:
    """Replace table-of-contents dot leaders with spaces of the same length (offsets stay valid)."""
    return _DOT_LEADER.sub(lambda match: " " * len(match.group(0)), text)


def is_toc_page(text: str) -> bool:
    """Detect table-of-contents style pages: many lines ending in dot leaders and a number."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    leader_lines = sum(1 for line in lines if re.search(r"(?:\.\s?){4,}\s*[\divxlcIVXLC]+\s*$", line))
    heading = re.search(r"\b(table\s+of\s+contents|contents)\b", text[:400], re.IGNORECASE) is not None
    return leader_lines >= 8 or (leader_lines >= 3 and (heading or leader_lines / len(lines) >= 0.3))


def _is_boundary(text: str, index: int) -> bool:
    """True when the punctuation at ``index`` ends a sentence."""
    char = text[index]
    if char not in ".!?":
        return False
    following = text[index + 1:index + 2]
    if following and not following.isspace():
        return False
    rest = text[index + 1:].lstrip()
    if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'(“‘["):
        return False
    if char == ".":
        word_start = index
        while word_start > 0 and (text[word_start - 1].isalpha() or text[word_start - 1] == "."):
            word_start -= 1
        word = text[word_start:index].lower()
        if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            return False
    return True


def split_sentences(text: str, max_chars: int = 500) -> list[tuple[int, int]]:
    """Split text into sentence spans; long spans are cut at line breaks, then at max_chars."""
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    index = 0
    while index < length:
        if text.startswith("\n", index) and re.match(r"\n[ \t]*\n", text[index:index + 64]):
            spans.append((start, index))
            start = index + 1
        elif _is_boundary(text, index):
            spans.append((start, index + 1))
            start = index + 1
        index += 1
    spans.append((start, length))

    result: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        for piece_start, piece_end in _limit_span(text, span_start, span_end, max_chars):
            # Trim surrounding whitespace so the quote is clean but still verbatim.
            while piece_start < piece_end and text[piece_start].isspace():
                piece_start += 1
            while piece_end > piece_start and text[piece_end - 1].isspace():
                piece_end -= 1
            if piece_end > piece_start:
                result.append((piece_start, piece_end))
    return result


def _limit_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_chars:
        cut = text.rfind("\n", cursor, cursor + max_chars)
        if cut <= cursor:
            cut = text.rfind(" ", cursor, cursor + max_chars)
        if cut <= cursor:
            cut = cursor + max_chars
        pieces.append((cursor, cut))
        cursor = cut
    pieces.append((cursor, end))
    return pieces


def clean_surface(name: str) -> str:
    """Join line-break hyphens, collapse whitespace, strip edge punctuation and determiners."""
    value = _LINE_BREAK_HYPHEN.sub(r"\1\2", name)
    value = re.sub(r"\s+", " ", value).strip(" .,;:()[]{}\"'“”‘’-–")
    words = value.split(" ")
    while words and words[0].lower() in _LEADING_DETERMINERS:
        words = words[1:]
    return " ".join(words).strip()


def singularize(word: str) -> str:
    """Very small plural-to-singular rule set for English nouns."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "ys", "as", "os")):
        return word[:-1]
    return word


def normalize_concept(name: str) -> str:
    """Normalized concept name: cleaned, lowercased, last word singular."""
    words = clean_surface(name).lower().split()
    if not words:
        return ""
    words[-1] = singularize(words[-1])
    return " ".join(words)


def concept_key(name: str) -> str:
    """Hash key used for alias lookups: every word singular, hyphens and spaces removed."""
    words = re.findall(r"[a-z0-9]+", normalize_concept(name).replace("-", ""))
    return "".join(singularize(word) for word in words)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
