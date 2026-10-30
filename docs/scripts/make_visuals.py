"""README 움짤 만들기 — 실제 노드 결과(mp4)로만 만든다. 글자는 넣지 않는다(한 · 영 README 가 같이 쓴다).

python docs/scripts/make_visuals.py <설정화 그림> <선화 영상 mp4> <채색 결과 mp4> <출력 gif> [칸 너비]
python docs/scripts/make_visuals.py thumb <선화 그림> <채색 결과 mp4> <출력 jpg>   — 워크플로우 썸네일(왼쪽 선화 | 오른쪽 채색 가운데 프레임)

[설정화 | 선화 영상 | 채색 결과] 세 칸을 나란히 놓는다. 설정화 칸은 멈춰 있고 두 영상 칸이 같은 프레임으로 움직인다.
필요한 것: Pillow · PyAV (플러그인 가상환경에 다 있다).
"""
import sys

import av
from PIL import Image, ImageOps

GAP = 6
BG = (255, 255, 255)
GAP_COLOR = (200, 200, 200)  # 흰 바탕 칸끼리 구분


def read_frames(path):
    with av.open(path) as c:
        stream = c.streams.video[0]
        fps = float(stream.average_rate or 12)
        return [f.to_image().convert("RGB") for f in c.decode(stream)], fps


def fit(img, w, h):
    return ImageOps.pad(img.convert("RGB"), (w, h), color=BG, method=Image.LANCZOS)


def center_square(img):
    """가로 그림의 가운데 정사각형(캐릭터가 가운데 있는 예제 그림용)."""
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def thumb(line_path, color_path, out_path, w=640, h=400):
    colors, _ = read_frames(color_path)
    with Image.open(line_path) as im:
        line = center_square(im.convert("RGB"))
    canvas = Image.new("RGB", (w + GAP, h), GAP_COLOR)
    canvas.paste(fit(line, w // 2, h), (0, 0))
    canvas.paste(fit(center_square(colors[len(colors) // 2]), w // 2, h), (w // 2 + GAP, 0))
    canvas.save(out_path, quality=90)
    print(out_path, canvas.size)


def main():
    if sys.argv[1] == "thumb":
        thumb(*sys.argv[2:5])
        return
    ref_path, line_path, color_path, out_path = sys.argv[1:5]
    cell_w = int(sys.argv[5]) if len(sys.argv) > 5 else 384
    cell_h = cell_w * 10 // 16
    with Image.open(ref_path) as im:
        ref = fit(ImageOps.exif_transpose(im), cell_w, cell_h)
    lines, fps = read_frames(line_path)
    colors, _ = read_frames(color_path)
    n = min(len(lines), len(colors))
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (cell_w * 3 + GAP * 2, cell_h), GAP_COLOR)
        canvas.paste(ref, (0, 0))
        canvas.paste(fit(lines[i], cell_w, cell_h), (cell_w + GAP, 0))
        canvas.paste(fit(colors[i], cell_w, cell_h), ((cell_w + GAP) * 2, 0))
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    duration = max(40, int(round(1000 / fps)))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True, disposal=1)
    print(out_path, n, "frames", f"{duration}ms")


if __name__ == "__main__":
    main()
