"""podsearch-bench: time searches over queries.txt."""

import argparse
import time

from podsearch.cli import common
from podsearch.core import index


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-bench",
        description="Time searches over queries.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries", type=int, default=5, help="number of queries from queries.txt to run"
    )
    parser.add_argument("--repeat", type=int, default=1, help="repetitions of the query set")
    parser.add_argument("-k", type=int, default=5, help="hits per query")
    parser.add_argument(
        "--impl", choices=["naive", "numpy"], default="naive", help="search implementation"
    )
    args = parser.parse_args()

    idx = index.load()
    model = index.load_model()
    queries = common.QUERIES_PATH.read_text().splitlines()[: args.queries]
    search = common.search_fn(args.impl)
    t0 = time.perf_counter()
    for _rep in range(args.repeat):
        for q in queries:
            search(idx, model, q, args.k)
    elapsed = time.perf_counter() - t0
    n = len(queries) * args.repeat
    print(f"{n} queries in {elapsed:.1f} s ({elapsed / n:.2f} s/query)")


if __name__ == "__main__":
    main()
