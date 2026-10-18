"""`python -m livetrans` 로 서버 실행. host/port 는 설정에서 읽는다."""

from __future__ import annotations

import logging

import uvicorn

from livetrans.core.config import load_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    uvicorn.run("livetrans.main:app", host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
