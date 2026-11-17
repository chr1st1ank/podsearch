"""Parse stored RSS bodies into episode dicts. Used by the ingest stage."""

import calendar
import re
from typing import Any

import feedparser_rs  # no manylinux cp315 wheel yet - builds from sdist (needs cargo)


def lang_code(language: str | None) -> str:
    """Normalize a catalog language value to a primary subtag; 'und' if missing."""
    tags = re.sub(r"[-_,;\s]+", " ", (language or "").lower()).split()
    return tags[0] if tags else "und"


def parse_feed(
    body: str,
    feed_url: str,
    podcast: str,
    max_episodes: int | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Parse one RSS body into episode dicts. Used by the ingest stage."""
    parsed = feedparser_rs.parse(body, sanitize_html=False, resolve_relative_uris=False)
    feed_title = parsed.feed.get("title")
    podcast_name = podcast or (str(feed_title) if feed_title else "")
    lang = lang_code(language)
    episodes: list[dict[str, Any]] = []
    entries = [e for e in parsed.entries if e.get("link")]  # unlinkable hits are useless: drop
    if max_episodes is not None:
        entries = entries[:max_episodes]
    for entry in entries:
        if entry.get("content"):
            description = entry.content[0].get("value", "")
        else:
            description = entry.get("summary") or entry.get("itunes_summary") or ""
        guid = entry.get("id") or entry["link"]
        parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
        episodes.append(
            {
                "feed_url": feed_url,
                "guid": guid,
                "podcast": podcast_name,
                "title": entry.get("title") or "(untitled)",
                "link": entry["link"],
                "pubdate": calendar.timegm(parsed_date) if parsed_date else 0,
                "description": description,
                "language": lang,
            }
        )
    return episodes
