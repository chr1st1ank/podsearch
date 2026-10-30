"""Parse stored RSS bodies into episode dicts. Used by the ingest stage."""

import calendar
from typing import Any, Literal

import feedparser  # TODO: kept as slow baseline for the profiling demo
import feedparser_rs  # TODO: no manylinux cp315 wheel yet - builds from sdist (needs cargo)

Parser = Literal["rs", "feedparser"]


def lang_code(language: str | None) -> str:
    """Normalize a catalog language value to a primary subtag; 'und' if missing."""
    return (language or "und").split("-")[0].strip().lower() or "und"


def parse_feed(
    body: str,
    feed_url: str,
    podcast: str,
    max_episodes: int | None = None,
    impl: Parser = "rs",
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Parse one RSS body into episode dicts. Used by the ingest stage."""
    # TODO: Bottleneck (feedparser impl). Flags lead to 1.65x speedup
    fp = feedparser_rs if impl == "rs" else feedparser
    parsed = fp.parse(body, sanitize_html=False, resolve_relative_uris=False)
    feed_title = parsed.feed.get("title")
    podcast_name = podcast or (str(feed_title) if feed_title else "")
    lang = lang_code(language)
    episodes: list[dict[str, Any]] = []
    entries = parsed.entries[:max_episodes] if max_episodes is not None else parsed.entries
    for entry in entries:
        if entry.get("content"):
            description = entry.content[0].get("value", "")
        else:
            description = entry.get("summary") or entry.get("itunes_summary") or ""
        guid = entry.get("id") or entry.get("link") or entry.get("title") or ""
        parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
        episodes.append(
            {
                "feed_url": feed_url,
                "guid": guid,
                "podcast": podcast_name,
                "title": entry.get("title") or "(untitled)",
                "link": entry.get("link") or "",
                "pubdate": calendar.timegm(parsed_date) if parsed_date else 0,
                "description": description,
                "language": lang,
            }
        )
    return episodes
