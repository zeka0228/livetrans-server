import pytest

from livetrans.api.capture import build_targets
from livetrans.plugins.source.loopback import parse_target

WINDOWS = [
    {"pid": 10, "process": "chrome.exe", "title": "방송 A - YouTube - Chrome"},
    {"pid": 20, "process": "Discord.exe", "title": "Discord"},
    {"pid": 10, "process": "chrome.exe", "title": "다른 창 - Chrome"},
]


def test_system_first_and_grouped_by_process():
    targets = build_targets(WINDOWS, app_capture=True)
    assert [t["id"] for t in targets] == ["system", "pid:10", "pid:20"]
    chrome = targets[1]
    assert chrome["title"] == "방송 A - YouTube - Chrome"
    assert chrome["windows"] == ["방송 A - YouTube - Chrome", "다른 창 - Chrome"]
    assert chrome["kind"] == "app" and chrome["process"] == "chrome.exe" and chrome["pid"] == 10


def test_without_app_capture_only_system():
    assert [t["id"] for t in build_targets(WINDOWS, app_capture=False)] == ["system"]


@pytest.mark.parametrize(
    ("target", "args"), [("system", ["system"]), ("pid:1234", ["pid", "1234"])]
)
def test_parse_target(target, args):
    assert parse_target(target) == args


@pytest.mark.parametrize("bad", ["", "pid:", "pid:abc", "chrome", "pid:-1"])
def test_parse_target_rejects(bad):
    with pytest.raises(ValueError):
        parse_target(bad)
