"""podsearch-query: run a single search."""

import argparse

from podsearch.cli import common
from podsearch.core import index


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-query",
        description="Run a single search against the index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("q", help="query text")
    parser.add_argument("-k", type=int, default=5, help="number of hits to show")
    parser.add_argument(
        "--impl", choices=["naive", "numpy"], default="naive", help="search implementation"
    )
    args = parser.parse_args()

    idx = index.load()
    model = index.load_model()
    hits = common.search_fn(args.impl)(idx, model, args.q, args.k)
    for hit in sorted(hits, key=lambda h: h.pubdate, reverse=True):
        print(common.fmt_hit(hit))


if __name__ == "__main__":
    main()
