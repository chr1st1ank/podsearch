"""Retrieval-quality regression tests against the real embedding model.

Downloads index.MODEL_NAME from Hugging Face on first run (cached afterwards).

ADR-0001 reverted to static model2vec embeddings for indexing speed and made
search language-scoped instead: cross-language ranking (the "Islandpferde"
compound-decomposition bug) is explicitly out of scope — a query only sees
same-language chunks. What remains worth guarding here is same-language
semantic recall: a query must still prefer the on-topic doc over superficially
similar distractors *within* a language shard.
"""

import numpy as np
import pytest

from podsearch.core import index

PASSAGES = {
    "de_horse": (
        "Pferde Gut, Alles Gut\n\nIn dieser Folge sprechen wir ausführlich über "
        "Pferde: die richtige Haltung, Fütterung und Pflege. Wir erklären, worauf "
        "man beim Sattelkauf achten sollte, warum Hufpflege so wichtig ist und wie "
        "man ein junges Pferd langsam an den Reitbetrieb gewöhnt. Außerdem gibt "
        "Jenny Tipps für den Stallalltag und erzählt von ihren Erfahrungen mit "
        "verschiedenen Rassen und Charakteren."
    ),
    "de_reifen": (
        "Automagazin\n\nReifen richtig wechseln: Worauf kommt es an? Wir "
        "erklären, wann Sommer- und Winterreifen montiert werden sollten, wie man "
        "das Profil misst und warum der richtige Luftdruck Sprit spart. Außerdem "
        "vergleichen wir aktuelle Modelle im Nassbrems-Test und sagen, welche "
        "Reifen sich für Vielfahrer lohnen."
    ),
    "de_kochen": (
        "Kochrunde\n\nHeute kochen wir ein schnelles Abendessen mit saisonalem "
        "Gemüse. Wir zeigen, wie man Zucchini, Paprika und Kartoffeln im Ofen "
        "perfekt gart, dazu gibt es einen selbstgemachten Kräuterquark. Außerdem "
        "diskutieren wir, welche Küchengeräte sich wirklich lohnen und wie man "
        "Reste clever verwertet, statt sie wegzuwerfen."
    ),
    "de_fussball": (
        "Kicker Talk\n\nFußball-Analyse: Wir besprechen das Meisterschaftsrennen, "
        "warum Pressing-Systeme den modernen Fußball dominieren und schauen auf "
        "die Top-Spiele des Wochenendes. Dazu ein Interview mit einem "
        "Jugendtrainer über die taktische Ausbildung junger Spieler."
    ),
}


@pytest.fixture(scope="module")
def model():
    return index.load_model()


def ranked(model, query: str) -> list[str]:
    """Passage keys best-first, scored exactly like the vector search."""
    q = np.asarray(model.encode([query])[0], dtype=np.float32)
    matrix = np.asarray(model.encode(list(PASSAGES.values())), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = matrix @ (q / np.linalg.norm(q))
    return [k for k, _ in sorted(zip(PASSAGES, sims, strict=True), key=lambda x: -x[1])]


def test_german_query_finds_german_doc(model) -> None:
    """ "Pferde" must rank the German horse doc first within the de shard."""
    assert ranked(model, "Pferde")[0] == "de_horse"


def test_same_language_queries_hit_on_topic_doc(model) -> None:
    """Same-language topical queries rank the matching doc first."""
    assert ranked(model, "Kochen")[0] == "de_kochen"
    assert ranked(model, "Fußball")[0] == "de_fussball"
