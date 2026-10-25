"""Stage 4: build the embedding index from the episodes corpus."""

import argparse
import time
from pathlib import Path

from podsearch.core import corpus, index
from podsearch.core.paths import CORPUS_DB, VECTORS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-index",
        description="Embed episode chunks into the float32 vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=CORPUS_DB, help="corpus SQLite database")
    parser.add_argument(
        "--vectors", type=Path, default=VECTORS_PATH, help="output file for raw float32 vectors"
    )
    parser.add_argument(
        "--max-chunks-per-episode",
        type=int,
        default=index.MAX_CHUNKS_PER_EPISODE,
        help="maximum text chunks embedded per episode",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    conn = corpus.open_corpus(args.db)
    count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    print(f"{count:,} episodes loaded")
    model = index.load_model()
    n = index.build_index(conn, model, args.vectors, args.max_chunks_per_episode)
    conn.close()
    size_mb = args.vectors.stat().st_size / 1e6
    print(
        f"{n:,} chunks -> {args.vectors} ({size_mb:.0f} MB, {time.perf_counter() - t0:.1f} s total)"
    )


if __name__ == "__main__":
    main()
