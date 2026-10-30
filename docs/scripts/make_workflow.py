"""예제 워크플로우(workflows/workflows.json) 만들기 — 팩 릴리스 zip 의 manifest 로 팩 노드 스냅숏을 채운다.

python docs/scripts/make_workflow.py <팩 zip> <팩 zip sha256> [색칠할 영상 종류 color|line]

이어진 선화 시작 · 끝(소스 이미지) → 팩 '두 프레임 → 중간 프레임' 12장 → 선화 채색(캐릭터 설정화는 소스 이미지).
노드 · 선 모양은 PromBoard 가 워크플로우를 저장하는 모양(WorkspaceNode · Wire) 그대로다 — promboard-app/yolo 와 같다.
"""
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACK_ID = "promboard.anime2d"
COUNT = 12  # 시작 + 12 + 끝 = 14프레임 — AniDoc 한 번에 칠하는 길이
SEED = 1  # 예제 그림으로 여러 시드를 돌려 색이 설정화와 가장 잘 맞은 값(땋은 머리 남색 · 셔츠 흰색 · 치마 하늘색)


def snapshot(plugin_id, node):
    snap = {"plugin": plugin_id, "id": node["id"], "label": node["label"]}
    # 앱이 보드에 저장하는 사본과 같은 칸(웹 toPluginNodeManifest) — 설명은 싣지 않는다.
    for key in ("template", "layout", "showSources", "sources", "settingsPanel", "views", "compare", "menuGroup", "inputs", "outputs", "settings"):
        if key in node:
            snap[key] = node[key]
    return snap


def plugin_node(node_id, x, y, plugin_id, node, settings):
    return {
        "id": node_id,
        "position": {"x": x, "y": y},
        "selectedTermIds": [],
        "isStale": False,
        "type": f"{plugin_id}/{node['id']}",
        "plugin": {"id": plugin_id, "nodeId": node["id"]},
        "snapshot": snapshot(plugin_id, node),
        "pluginSettings": settings,
        "nodeType": f"{plugin_id}/{node['id']}",
        "inputWires": [],
        "outputWires": [],
        "outputPick": {},
    }


def source_image(node_id, x, y):
    return {
        "id": node_id,
        "position": {"x": x, "y": y},
        "selectedTermIds": [],
        "isStale": False,
        "type": "sourceImage",
        "images": [],
        "nodeType": "sourceImage",
        "pendingUploads": [],
        "inputWires": [],
        "outputWires": [],
    }


def wire(wid, a, ap, b, bp, kind):
    return {"id": wid, "fromNodeId": a, "fromPortId": ap, "toNodeId": b, "toPortId": bp, "dataType": kind}


def main():
    pack_zip, pack_sha = sys.argv[1:3]
    source = sys.argv[3] if len(sys.argv) > 3 else "color"
    with zipfile.ZipFile(pack_zip) as z:
        pack = json.loads(z.read("manifest.json").decode("utf-8"))
    with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as fh:
        own = json.load(fh)
    fill = next(n for n in pack["nodes"] if n["id"] == "fill")
    colorize = next(n for n in own["nodes"] if n["id"] == "colorize")
    fill_settings = {s["id"]: s["default"] for s in fill["settings"]}
    fill_settings.update(mode="inbetween", count=COUNT)
    color_settings = {s["id"]: s["default"] for s in colorize["settings"]}
    color_settings.update(source=source, seed=SEED)
    workflow = {
        "id": "line-inbetween-colorize",
        "name": "이어진 선화 두 장 → 중간 프레임 → 선화 채색",
        "description": (
            "에셋 탭의 예제 카드를 소스 노드 위에 끌어 놓는다: '이어진 선화 시작 프레임' · '이어진 선화 끝 프레임' · '캐릭터 설정화'."
            " 그다음 '프레임 사이 채우기' 노드 ▷(14프레임 선화 영상), '선화 채색' 노드 ▷ 를 차례로 누른다(시드 1, 바꾸면 색이 달라진다)."
            " 첫 노드는 팩 카드라 없으면 '누락된 플러그인 설치'로 받는다. 내 선화 영상이 있으면 소스 영상 노드를 '선화 채색'에 바로 이어도 된다."
        ),
        "thumbnail": "workflows/line-inbetween-colorize.jpg",
        "nodes": [
            source_image("line-start", 0, 0),
            source_image("line-end", 0, 340),
            source_image("sheet", 380, 560),
            plugin_node("fill", 380, 60, PACK_ID, fill, fill_settings),
            plugin_node("colorize", 820, 200, own["id"], colorize, color_settings),
        ],
        "wires": [
            wire("start-to-fill", "line-start", "source-image-out", "fill", "in-start", "image"),
            wire("end-to-fill", "line-end", "source-image-out", "fill", "in-end", "image"),
            wire("fill-to-colorize", "fill", "out-video", "colorize", "in-video", "video"),
            wire("sheet-to-colorize", "sheet", "source-image-out", "colorize", "in-reference", "image"),
        ],
        "plugins": [{"id": PACK_ID, "version": pack["version"], "sha256": pack_sha}],
    }
    assert len(workflow["description"]) <= 300, len(workflow["description"])
    out = os.path.join(ROOT, "workflows", "workflows.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"schema": 1, "workflows": [workflow]}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(out, len(workflow["description"]), "chars")


if __name__ == "__main__":
    main()
