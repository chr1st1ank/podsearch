"""podsearch CLI: query / bench."""

import argparse
import time
from pathlib import Path

from podsearch.core import index

QUERIES_PATH = Path(__file__).resolve().parent / "queries.txt"


def _fmt(h: index.Hit) -> str:
    return f"  {h.score:.4f}  {h.podcast} — {h.episode}"


def _search(args):
    return index.search_numpy if args.impl == "numpy" else index.search


def cmd_query(args) -> None:
    idx = index.load()
    model = index.load_model()
    for hit in _search(args)(idx, model, args.q, args.k):
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
    parser = argparse.ArgumentParser(prog="podsearch")
    sub = parser.add_subparsers(required=True)
    p_q = sub.add_parser("query")
    p_q.add_argument("q")
    p_q.add_argument("-k", type=int, default=5)
    p_q.add_argument("--impl", choices=["naive", "numpy"], default="naive")
    p_q.set_defaults(func=cmd_query)
    p_b = sub.add_parser("bench")
    p_b.add_argument("--queries", type=int, default=5)
    p_b.add_argument("--repeat", type=int, default=1)
    p_b.add_argument("-k", type=int, default=5)
    p_b.add_argument("--impl", choices=["naive", "numpy"], default="naive")
    p_b.set_defaults(func=cmd_bench)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
