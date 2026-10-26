"""podsearch-serve: run the FastAPI search service on :8000."""

import uvicorn

from podsearch.api import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
