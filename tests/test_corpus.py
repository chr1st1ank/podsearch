"""Tests for pepsearch.core.corpus."""

from pepsearch.core import corpus


def test_clean_strips_rst_roles() -> None:
    text = "PEP 8\nTitle: Test\n\nAbstract:\n  See :pep:`8` and ``inline code``.\n"
    cleaned = corpus.clean(text)
    assert "PEP 8" in cleaned
    assert ":pep:" not in cleaned
    assert "inline code" in cleaned


def test_clean_strips_directives() -> None:
    text = "Intro.\n\n.. code-block:: python\n\n    x = 1\n"
    assert "code-block" not in corpus.clean(text)


def test_chunk_word_windows() -> None:
    words = " ".join(f"w{i}" for i in range(500))
    chunks = corpus.chunk(words, size=200, overlap=40)
    assert len(chunks) == 4
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w160"
    assert chunks[0].split()[-1] == chunks[1].split()[39]


def test_chunk_short_text() -> None:
    assert corpus.chunk("one two three") == ["one two three"]
