"""Tests for podsearch.core.parse - RSS body -> episode dicts."""

from podsearch.core import parse

RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Feed Title</title>
    <item>
      <title>Episode One</title>
      <link>https://example.com/1</link>
      <guid>guid-1</guid>
      <pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate>
      <description>&lt;p&gt;Shownotes &lt;b&gt;here&lt;/b&gt;&lt;/p&gt;</description>
      <enclosure url="https://example.com/1.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode Two</title>
      <guid>guid-2</guid>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_entries() -> None:
    eps = parse.parse_feed(RSS.decode(), "https://feed", "Feed Title")
    assert len(eps) == 2
    first = eps[0]
    assert first["feed_url"] == "https://feed"
    assert first["guid"] == "guid-1"
    assert first["title"] == "Episode One"
    assert first["link"] == "https://example.com/1"
    assert first["pubdate"] == 1_704_067_200
    assert "<p>" in first["description"]  # stripped later, at ingest


def test_parse_feed_max_episodes() -> None:
    assert len(parse.parse_feed(RSS.decode(), "u", "p", max_episodes=1)) == 1


def test_parse_feed_rs_matches_feedparser() -> None:
    args = (RSS.decode(), "https://feed", "Feed Title")
    assert parse.parse_feed(*args, impl="rs") == parse.parse_feed(*args, impl="feedparser")


def test_lang_code() -> None:
    assert parse.lang_code("de-AT") == "de"
    assert parse.lang_code(" EN ") == "en"
    assert parse.lang_code("en_US") == "en"
    assert parse.lang_code("de_de") == "de"
    assert parse.lang_code("en,es") == "en"
    assert parse.lang_code("") == "und"
    assert parse.lang_code(None) == "und"
