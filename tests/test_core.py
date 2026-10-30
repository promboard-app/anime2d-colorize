"""모델 없이 도는 순수 함수 시험 — `python -m pytest tests` (플러그인 폴더에서).

설치 zip 에는 들어가지 않는다(.gitattributes export-ignore).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import main  # noqa: E402


# ── 14프레임 묶기 (Review Focus 5) ──────────────────────────────────────────


def test_chunk14_short_video_repeats_last_frame():
    parts = main.chunk14(list(range(5)))
    assert parts == [[0, 1, 2, 3, 4] + [4] * 9]
    assert main.stitch([[f"c{x}" for x in p] for p in parts], 5) == ["c0", "c1", "c2", "c3", "c4"]


def test_chunk14_exact_and_remainder():
    assert main.chunk14(list(range(14))) == [list(range(14))]
    parts = main.chunk14(list(range(30)))
    assert [len(p) for p in parts] == [14, 14, 14]
    assert parts[2] == [28, 29] + [29] * 12
    assert main.stitch(parts, 30) == list(range(30))


def test_chunk14_single_frame():
    parts = main.chunk14(["a"])
    assert parts == [["a"] * 14]
    assert main.stitch(parts, 1) == ["a"]


# ── 입력 검사 ────────────────────────────────────────────────────────────────


def test_check_inputs_ok():
    assert main.check_inputs({"video": "v.mp4", "reference": "r.png"}) is None


@pytest.mark.parametrize(
    "inputs, now, card",
    [
        ({}, "지금은 아무것도 이어져 있지 않습니다", "예제 — 캐릭터 설정화"),
        ({"video": "v.mp4"}, "지금은 '색칠할 영상'만 이어져 있습니다", "예제 — 캐릭터 설정화"),
        ({"reference": "r.png"}, "지금은 '캐릭터 설정화'만 이어져 있습니다", "이어진 선화 두 장"),
    ],
)
def test_check_inputs_names_what_is_connected(inputs, now, card):
    msg = main.check_inputs(inputs)
    assert msg.startswith("'선화 채색'은 '색칠할 영상'과 '캐릭터 설정화'를 이어야 합니다.")
    assert now in msg
    assert card in msg


def test_check_length():
    assert main.check_length(main.MAX_FRAMES, 24) is None
    msg = main.check_length(240, 24, max_frames=98)
    assert msg == "이 영상은 4초까지 받습니다(98프레임). 지금 영상은 10초입니다. 앞부분만 잘라 이어 주세요."
    assert "초까지 받습니다" in main.check_length(200, 0, max_frames=98)  # fps 를 모르면 24 로 본다


# ── 비율 맞추기 ──────────────────────────────────────────────────────────────


def test_fit_box_keeps_ratio_inside_512x320():
    assert main.fit_box(1024, 640) == (512, 320, 0, 0)
    w, h, left, top = main.fit_box(1920, 1080)  # 16:9 → 위아래 여백
    assert (w, h) == (512, 288) and (left, top) == (0, 16)
    w, h, left, top = main.fit_box(1080, 1920)  # 세로 영상 → 좌우 여백
    assert h == 320 and w == 180 and left == (512 - 180) // 2 and top == 0
    w, h, _, _ = main.fit_box(4000, 4000)
    assert (w, h) == (320, 320)
    assert all(v % 2 == 0 for v in main.fit_box(333, 777)[:2])


# ── 선화 그대로 쓰기 ────────────────────────────────────────────────────────


def test_line_to_sketch_binarizes_and_inverts_dark_background():
    np = pytest.importorskip("numpy")
    light = np.full((4, 4), 255, dtype=np.uint8)
    light[1, 1] = 0
    light[2, 2] = 180  # 흐린 선도 선(공식 기준 200 이하)
    out = main.line_to_sketch(light)
    assert out[1, 1] == 0 and out[2, 2] == 0 and out[0, 0] == 255
    dark = 255 - light  # 검은 바탕 흰 선
    assert (main.line_to_sketch(dark) == out).all()


# ── 쉬운 오류 문구 (Review Focus 4) ──────────────────────────────────────────


class OutOfMemoryError(RuntimeError):
    pass


def test_is_oom():
    assert main.is_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert main.is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not main.is_oom(ValueError("bad input"))


def test_messages_are_plain_korean():
    msg = main.oom_message(12282)
    assert msg.startswith("그래픽카드 메모리가 모자랍니다(선화 채색 약 8 GB 필요, 이 PC 12 GB).")
    assert "Error" not in msg and "CUDA out of memory" not in msg
    assert "알 수 없음" in main.oom_message(0)
    assert "NVIDIA 그래픽카드" in main.cpu_message()


def test_nodes_matches_manifest():
    import json

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert set(main.NODES) == {n["id"] for n in manifest["nodes"]}
