"""podsearch-serve: run the FastAPI search service on :8000."""

import argparse

import uvicorn

from podsearch import api
from podsearch.cli import common


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-serve",
        description="Run the FastAPI search service on :8000.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--impl", choices=common.IMPLS, default="naive", help="search implementation"
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rerank candidates with the cross-encoder",
    )
    args = parser.parse_args()
    uvicorn.run(api.create_app(args.impl, args.rerank), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
