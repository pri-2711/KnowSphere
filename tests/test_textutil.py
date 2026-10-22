"""Tokenizing, stemming, sentence spans and concept name keys."""

from src.textutil import (concept_key, is_toc_page, mask_dot_leaders, normalize_concept, split_sentences,
                          tokenize)


def test_sentence_spans_are_verbatim_and_respect_abbreviations() -> None:
    text = "Systems engineering is a method. It uses e.g. models and Fig. 3 data.\n\nNew paragraph here"
    spans = split_sentences(text)
    sentences = [text[start:end] for start, end in spans]
    assert sentences == ["Systems engineering is a method.", "It uses e.g. models and Fig. 3 data.",
                         "New paragraph here"]
    for (start, end), sentence in zip(spans, sentences):
        assert text[start:end] == sentence


def test_long_sentences_are_split_into_verbatim_pieces() -> None:
    text = ("word " * 300).strip()
    spans = split_sentences(text, max_chars=200)
    assert len(spans) > 1
    assert all(end - start <= 200 for start, end in spans)
    assert all(text[start:end] in text for start, end in spans)


def test_tokenize_lowercases_removes_stopwords_and_stems() -> None:
    assert tokenize("The Requirements were analysed by the engineers") == ["requir", "analys", "engin"]
    assert tokenize("requirement requirements") == ["requir", "requir"]


def test_concept_keys_fold_case_plural_hyphen_and_spacing() -> None:
    assert concept_key("Sub-Systems") == concept_key("sub system") == concept_key("subsystem")
    assert concept_key("Design Reviews") == concept_key("the design review")
    assert normalize_concept("the Design Reviews.") == "design review"
    assert concept_key("design review") != concept_key("preliminary design review")


def test_table_of_contents_detection_and_dot_leader_masking() -> None:
    toc = "Contents\nChapter 1 Intro ........ 1\nChapter 2 Methods ........ 5\nChapter 3 Results ....... 9\n"
    assert is_toc_page(toc)
    assert not is_toc_page("A normal page. It has sentences. Nothing else.")
    masked = mask_dot_leaders(toc)
    assert len(masked) == len(toc)
    assert "...." not in masked
