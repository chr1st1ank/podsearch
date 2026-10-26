"""Tiny threaded load generator for the FastAPI endpoint."""

import argparse
import statistics
import threading
import time

import httpx

from podsearch.cli.common import QUERIES_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-load",
        description="Threaded load generator for the search endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/search", help="endpoint to hit")
    parser.add_argument("--duration", type=float, default=30, help="load duration in seconds")
    parser.add_argument("--concurrency", type=int, default=4, help="worker threads")
    args = parser.parse_args()

    queries = [q for q in QUERIES_PATH.read_text().splitlines() if q.strip()]
    latencies: list[float] = []
    stop_at = time.perf_counter() + args.duration
    lock = threading.Lock()

    def worker(tid: int) -> None:
        client = httpx.Client(timeout=30)
        i = tid
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            r = client.get(args.url, params={"q": queries[i % len(queries)]})
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                with lock:
                    latencies.append(ms)
            i += len(queries)
        client.close()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(args.concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    latencies.sort()
    if not latencies:
        print("no successful requests")
        return
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(
        f"{len(latencies)} requests in {elapsed:.0f} s "
        f"({len(latencies) / elapsed:.2f} req/s)  p50={p50:.0f} ms p95={p95:.0f} ms "
        f"mean={statistics.mean(latencies):.0f} ms"
    )


if __name__ == "__main__":
    main()
