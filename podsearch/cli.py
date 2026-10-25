"""podsearch CLI: query / bench."""

import argparse
import time
from pathlib import Path

from podsearch.core import index

QUERIES_PATH = Path(__file__).resolve().parent / "queries.txt"


def _fmt(h: index.Hit) -> str:
    date = time.strftime("%Y-%m-%d", time.gmtime(h.pubdate)) if h.pubdate else "----------"
    return f"  {h.score:.4f}  {date}  {h.podcast} — {h.episode}"


def _search(args):
    return index.search_numpy if args.impl == "numpy" else index.search


def cmd_query(args) -> None:
    idx = index.load()
    model = index.load_model()
    hits = _search(args)(idx, model, args.q, args.k)
    for hit in sorted(hits, key=lambda h: h.pubdate, reverse=True):
        print(_fmt(hit))


def cmd_bench(args) -> None:
    idx = index.load()
    model = index.load_model()
    queries = QUERIES_PATH.read_text().splitlines()[: args.queries]
    search = _search(args)
    t0 = time.perf_counter()
    for _rep in range(args.repeat):
        for q in queries:
            search(idx, model, q, args.k)
    elapsed = time.perf_counter() - t0
    n = len(queries) * args.repeat
    print(f"{n} queries in {elapsed:.1f} s ({elapsed / n:.2f} s/query)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch",
        description="Semantic search over podcast episodes (local model2vec embeddings).",
    )
    sub = parser.add_subparsers(required=True)
    p_q = sub.add_parser(
        "query",
        help="run a single search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_q.add_argument("q", help="query text")
    p_q.add_argument("-k", type=int, default=5, help="number of hits to show")
    p_q.add_argument(
        "--impl", choices=["naive", "numpy"], default="naive", help="search implementation"
    )
    p_q.set_defaults(func=cmd_query)
    p_b = sub.add_parser(
        "bench",
        help="time searches over queries.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_b.add_argument(
        "--queries", type=int, default=5, help="number of queries from queries.txt to run"
    )
    p_b.add_argument("--repeat", type=int, default=1, help="repetitions of the query set")
    p_b.add_argument("-k", type=int, default=5, help="hits per query")
    p_b.add_argument(
        "--impl", choices=["naive", "numpy"], default="naive", help="search implementation"
    )
    p_b.set_defaults(func=cmd_bench)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
