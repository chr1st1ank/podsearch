"""Retrieval-quality regression tests against the real embedding model.

Downloads index.MODEL_NAME from Hugging Face on first run (cached afterwards).

These guard the failure modes of the old model2vec encoder (static
bag-of-subword embeddings): German compounds like "Islandpferde" decompose
into "Island"+fragments, so island episodes outranked Icelandic-horse
episodes, and cross-language queries favoured same-language documents.

Not asserted here, verified empirically: "Horses" returning only English
hits is a scale effect that a micro corpus cannot reproduce, and
"Reiten"->"Reifen" ranks wrong even under e5.
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
        "verschiedenen Rassen und Charakteren. Ein Muss für alle, die mit Pferden "
        "arbeiten oder einfach mehr über sie lernen wollen."
    ),
    "en_horse": (
        "Horse Radio Network\n\nToday we talk about horses: proper stabling, "
        "feeding and daily care. We explain what to look for when buying a saddle, "
        "why hoof care matters so much and how to slowly introduce a young horse "
        "to regular riding work. Jenny also shares practical tips for barn "
        "routines and stories from her experience with different breeds and "
        "temperaments. A must-listen for everyone who works with horses or simply "
        "wants to learn more about them."
    ),
    "en_icelandic": (
        "Horseman's Corner\n\nIcelandic horses are a remarkable breed: small, "
        "sturdy and famous for their smooth tolt gait. In this episode we discuss "
        "the history of these hardy animals, how they adapted to the harsh "
        "climate of Iceland, and what riders should know about their unique five "
        "gaits. We also cover buying an icelandic horse, import rules, and why "
        "their sure-footedness makes them perfect trail companions."
    ),
    "en_island": (
        "The Incredible Journey\n\nThe secrets of the world's most beautiful "
        "islands: sandy beaches, hidden coves and ferry routes that connect "
        "remote coastal communities. This week we travel to a tropical island "
        "paradise, explore how island ecosystems develop, and hear from locals "
        "about life at the edge of the sea. Plus: the best island-hopping "
        "itineraries for your next summer vacation."
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
    "en_tech": (
        "Daily Tech\n\nThis week in technology: we review the latest laptop "
        "releases, discuss whether folding phones are finally ready for the "
        "mainstream, and explain what the new chipset announcements mean for "
        "battery life. Plus: our favorite productivity apps of the month and a "
        "look at upcoming software updates."
    ),
    "en_soccer": (
        "Football Weekly\n\nSoccer tactics, transfer news and match analysis: "
        "this week we break down the championship race, discuss why pressing "
        "systems dominate modern football, and preview the weekend's biggest "
        "fixtures. Plus an interview with a youth coach about developing "
        "technical skills in young players."
    ),
    "de_fussball": (
        "Kicker Talk\n\nFußball-Analyse: Wir besprechen das Meisterschaftsrennen, "
        "warum Pressing-Systeme den modernen Fußball dominieren und schauen auf "
        "die Top-Spiele des Wochenendes. Dazu ein Interview mit einem "
        "Jugendtrainer über die technische Ausbildung junger Spieler."
    ),
    "en_cook": (
        "Kitchen Talk\n\nToday we cook a quick weeknight dinner with seasonal "
        "vegetables. We show you how to roast zucchini, peppers and potatoes to "
        "perfection, plus a homemade herb quark dip. We also discuss which "
        "kitchen gadgets are actually worth buying and how to use up leftovers "
        "cleverly."
    ),
}


@pytest.fixture(scope="module")
def model():
    return index.load_model()


def ranked(model, query: str) -> list[str]:
    """Passage keys best-first, scored exactly like the vector search."""
    encode_query = getattr(model, "encode_query", model.encode)
    q = np.asarray(encode_query([query])[0], dtype=np.float32)
    matrix = np.asarray(model.encode(list(PASSAGES.values())), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = matrix @ (q / np.linalg.norm(q))
    return [k for k, _ in sorted(zip(PASSAGES, sims, strict=True), key=lambda x: -x[1])]


def test_compound_query_matches_semantics_not_subwords(model) -> None:
    """ "Islandpferde" must find the (English) Icelandic-horses doc, not islands.

    model2vec splits the compound into "Island"+fragments and ranks the
    islands travel doc first (0.50 vs 0.23); e5 gets it right.
    """
    assert ranked(model, "Islandpferde")[0] == "en_icelandic"


def test_german_query_finds_german_doc(model) -> None:
    """Sanity guard: the German horse doc stays top-1 for "Pferde"."""
    assert ranked(model, "Pferde")[0] == "de_horse"
