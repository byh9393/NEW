import base64
import hashlib
import hmac
import json
import re

from upbit_bot.trading.executor import _jwt_token


def _b64decode(segment: str) -> bytes:
    padding = '=' * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def test_jwt_token_is_urlsafe_and_signed():
    access_key = "test-access"
    secret_key = "secret-key"
    query = {"market": "KRW-BTC", "price": 123.45}

    token = _jwt_token(access_key, secret_key, query)
    header_b64, payload_b64, signature_b64 = token.split(".")

    # URL-safe Base64 문자열은 '+', '/', '=' 문자를 포함하지 않는다.
    assert not re.search(r"[+/=]", token)

    header = json.loads(_b64decode(header_b64))
    payload = json.loads(_b64decode(payload_b64))

    assert header == {"typ": "JWT", "alg": "HS256"}
    assert payload["access_key"] == access_key
    assert payload["query_hash_alg"] == "SHA512"
    assert payload["nonce"].isdigit()

    expected_query = json.dumps(query, separators=(",", ":"), ensure_ascii=False)
    expected_hash = hashlib.sha512(expected_query.encode()).hexdigest()
    assert payload["query_hash"] == expected_hash

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_signature = hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest()
    expected_signature_b64 = base64.urlsafe_b64encode(expected_signature).decode().rstrip("=")
    assert signature_b64 == expected_signature_b64
