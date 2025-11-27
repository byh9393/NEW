"""
upbit_bot 패키지 초기화.

- `.env` 파일을 자동으로 로드해 OpenAI와 업비트 API 키를 환경 변수로 주입한다.
- 하위 모듈이 별도 설정 없이 `os.environ`을 통해 키를 읽을 수 있게 한다.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# 현재 작업 디렉터리(.env 우선)와 저장소 루트의 .env 모두 시도
load_dotenv()
root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=root_env)
