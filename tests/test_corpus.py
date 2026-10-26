"""Tests for podsearch.core.corpus."""

from podsearch.core import corpus


def test_strip_html_removes_tags() -> None:
    assert corpus.strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_unescapes_entities() -> None:
    assert corpus.strip_html("fish &amp; chips &lt;3") == "fish & chips <3"


def test_strip_html_collapses_whitespace() -> None:
    assert corpus.strip_html("a\n\n  b\t\tc") == "a b c"


def test_chunk_word_windows() -> None:
    words = " ".join(f"w{i}" for i in range(500))
    chunks = corpus.chunk(words, size=200, overlap=40)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w160"
    assert chunks[2].split()[0] == "w320"
    assert chunks[2].split()[-1] == "w499"
    assert chunks[0].split()[-1] == chunks[1].split()[39]


def test_chunk_folds_sliver_tail() -> None:
    # a 210-word text would otherwise yield a 50-word tail chunk
    words = " ".join(f"w{i}" for i in range(210))
    chunks = corpus.chunk(words, size=200, overlap=40)
    assert chunks == [words]


def test_chunk_short_text() -> None:
    assert corpus.chunk("one two three") == ["one two three"]
