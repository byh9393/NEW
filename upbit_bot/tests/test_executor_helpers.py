import pytest

from upbit_bot.trading.executor import _compute_throttle_delay, _parse_remaining_req


def test_parse_remaining_req_ignores_non_numeric_fields():
    header = "group=market; min=59; sec=5"
    parsed = _parse_remaining_req(header)

    assert parsed["min"] == 59
    assert parsed["sec"] == 5
    assert "group" not in parsed  # 문자열 값은 무시


@pytest.mark.parametrize(
    "header,expected_delay",
    [
        ("group=market; min=5; sec=5", 0.0),
        ("group=market; min=1; sec=0", 1.0),
        ("group=market; min=0; sec=2", 1.0),
    ],
)
def test_compute_throttle_delay(header: str, expected_delay: float):
    headers = {"Remaining-Req": header}
    delay = _compute_throttle_delay(headers)

    assert delay == pytest.approx(expected_delay, rel=0.1)

