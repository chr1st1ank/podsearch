"""Indexing entrypoint: build the embedding index from the PEP corpus."""

import time

from pepsearch.core import corpus, index


def main() -> None:
    t0 = time.perf_counter()
    peps = corpus.load_peps()
    print(f"{len(peps)} PEPs loaded ({time.perf_counter() - t0:.1f} s)")
    model = index.load_model()
    idx = index.build_index(peps, model)
    index.save(idx)
    print(
        f"{len(idx.chunks)} chunks, dim={len(idx.vectors[0])}, "
        f"saved to {index.INDEX_PATH} ({time.perf_counter() - t0:.1f} s total)"
    )


if __name__ == "__main__":
    main()
