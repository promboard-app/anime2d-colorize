"""promboard.anime2d-colorize — 2D 애니메이션 팩 · 선화 채색 노드 코드(AniDoc, CVPR 2025).

노드 하나(colorize)가 색칠할 영상과 캐릭터 설정화 1장을 받아, 설정화의 색으로 칠한 영상(512×320 안에 맞춤)과 그 프레임을 낸다.
  선화 영상(source=line)   흰 바탕 검은 선 영상을 그대로 칠한다 — 공식 코드의 선화 추출을 건너뛴다(저자 답변, AniDoc 이슈 #17).
  컬러 영상(source=color)  공식 코드처럼 프레임마다 선화를 뽑아(informative-drawings sk_model.pth) 칠한다.
AniDoc 은 정확히 14프레임을 칠한다. 긴 영상은 14프레임씩 나눠 칠하고 잇는다(마지막 묶음은 끝 프레임을 반복해 채우고 결과에서 뺀다).
묶음마다 공식 추론(scripts_infer/anidoc_inference.sh: --all_sketch --matching --tracking --max_point 10)과 같은 순서로 조건을 만든다:
  설정화와 첫 선화의 점 맞추기(SuperPoint + LightGlue) → 그 점을 14프레임 동안 따라가기(CoTracker2) → 점 지도 + 설정화 + 선화 → ControlNet.
생성: AniDocPipeline(SVD img2vid-xt 미세조정, fp16, 25단계, 공식 값 motion_bucket_id 127 · fps 7 · noise_aug 0.02 · decode_chunk 8).
  그래픽카드 메모리가 FULL_GPU_MB 보다 작으면 diffusers 모델 CPU 오프로드(지금 쓰는 부품만 그래픽카드에 올림)로 돈다.
그림 비율은 흰 여백으로 512×320 에 맞춰 칠하고, 결과에서 여백을 잘라 입력 비율로 돌려준다(공식 코드는 늘려 맞춘다).

상류 코드는 nodes/anidoc · nodes/cotracker · nodes/lightglue 에 최소로 고쳐 담았다(고친 곳은 `# PromBoard:` 주석, 목록은 NOTICE.md).
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANIDOC_DIR = os.path.join(HERE, "anidoc")

SVD = "stabilityai/stable-video-diffusion-img2vid-xt"
ANIDOC = "Yhmeng1106/anidoc"
SK_MODEL = "lllyasviel/Annotators/sk_model.pth"
COTRACKER = "facebook/cotracker/cotracker2.pth"
SUPERPOINT = "lightglue/superpoint_v1.pth"  # cvg/LightGlue GitHub 릴리스 v0.1_arxiv
LIGHTGLUE = "lightglue/superpoint_lightglue.pth"

WIDTH, HEIGHT = 512, 320
CHUNK = 14  # AniDoc 이 한 번에 칠하는 프레임 수
MAX_FRAMES = 240  # 길이 상한(24fps 10초) — RTX 4070 SUPER 12GB 실측 14프레임 약 21초 → 약 6분. 2배 느린 카드까지 900초 안(더 느리면 첫 묶음 뒤 어림으로 멈춘다)
MAX_FRAME_OUTPUTS = 64  # 여러 장 출력 포트 상한(02 §3)
NEED_MB = 8192  # 실측: 오프로드로 최고 할당 6.6GB, 7GB 로 묶어도 돈다 — 메모리 부족 안내에 쓴다
FULL_GPU_MB = 15360  # 이보다 큰 그래픽카드는 오프로드 없이 통째로 올린다(최고 할당 8.9GB — 12GB 에서는 넘쳐 오히려 느렸다)
TIME_BUDGET_SEC = 1050  # PromBoard 실행기 노드 한 번 900초 + 적재 몫 180초(조금 남긴다)
STEPS = 25
DEVICE = "cuda"  # 그래픽카드 전용(ctx.device 가 cuda 가 아니면 모델 전에 멈춘다). 시험대만 cpu 로 바꾼다
LINE_THRESHOLD = 200  # 공식 코드의 선화 양자화 기준(sketch > 200 → 흰 바탕)

LABEL_NODE = "선화 채색"
LABEL_VIDEO = "색칠할 영상"
LABEL_REFERENCE = "캐릭터 설정화"
CARD_REFERENCE = "예제 — 캐릭터 설정화"
WORKFLOW_NAME = "이어진 선화 두 장 → 중간 프레임 → 선화 채색"

_CACHE = {"pipe": None, "offload": None, "aux": None}


# ── 모델 없이 시험하는 순수 함수 ────────────────────────────────────────────


def check_inputs(inputs):
    """입력 조합 검사 — 모델을 불러오기 전에. 잘못이면 한국어 문구, 맞으면 None."""
    has_video, has_ref = bool(inputs.get("video")), bool(inputs.get("reference"))
    if has_video and has_ref:
        return None
    if has_video:
        now = f"지금은 '{LABEL_VIDEO}'만 이어져 있습니다"
    elif has_ref:
        now = f"지금은 '{LABEL_REFERENCE}'만 이어져 있습니다"
    else:
        now = "지금은 아무것도 이어져 있지 않습니다"
    hint = (
        f"에셋의 '{CARD_REFERENCE}' 카드를 소스 노드 위에 끌어 놓아 보세요."
        if not has_ref
        else f"선화 영상이 없으면 예제 워크플로우 '{WORKFLOW_NAME}'로 만들어 보세요."
    )
    return f"'{LABEL_NODE}'은 '{LABEL_VIDEO}'과 '{LABEL_REFERENCE}'를 이어야 합니다. {now}. {hint}"


def check_length(frame_count, fps, max_frames=MAX_FRAMES):
    """길이 검사. frame_count 가 상한을 넘으면 'N초까지 받습니다. 지금 영상은 M초입니다.' 문구, 아니면 None."""
    if frame_count <= max_frames:
        return None
    rate = fps if fps and fps > 0 else 24.0
    limit = math.floor(max_frames / rate * 10) / 10
    now = math.ceil(frame_count / rate * 10) / 10
    return (
        f"이 영상은 {_num(limit)}초까지 받습니다({max_frames}프레임). 지금 영상은 {_num(now)}초입니다."
        " 앞부분만 잘라 이어 주세요."
    )


def _num(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


def chunk14(frames, size=CHUNK):
    """프레임 목록 → size 장씩 묶은 목록. 마지막 묶음이 모자라면 끝 프레임을 반복해 채운다."""
    frames = list(frames)
    out = []
    for i in range(0, len(frames), size):
        part = frames[i : i + size]
        out.append(part + [part[-1]] * (size - len(part)))
    return out


def stitch(parts, count):
    """묶음마다 칠한 결과를 잇고 원래 프레임 수로 자른다(채운 끝 프레임을 뺀다)."""
    return [f for part in parts for f in part][:count]


def fit_box(width, height, box_w=WIDTH, box_h=HEIGHT):
    """(width, height) 그림을 비율 그대로 box 안에 넣을 크기와 자리 → (새 너비, 새 높이, 왼쪽, 위). 크기는 짝수(H.264)."""
    scale = min(box_w / width, box_h / height)
    w = max(2, min(box_w, int(round(width * scale))) // 2 * 2)
    h = max(2, min(box_h, int(round(height * scale))) // 2 * 2)
    return w, h, (box_w - w) // 2, (box_h - h) // 2


def line_to_sketch(gray):
    """선화 프레임(회색조 uint8 (h, w)) → AniDoc 선화 조건(흰 바탕 255 · 검은 선 0). 검은 바탕에 흰 선이면 뒤집는다."""
    import numpy as np

    gray = np.asarray(gray, dtype=np.uint8)
    if gray.mean() < 128:
        gray = 255 - gray
    return ((gray > LINE_THRESHOLD).astype(np.uint8) * 255)


def is_oom(exc):
    """그래픽카드 메모리 부족 예외인가(torch.OutOfMemoryError · 'CUDA out of memory' · cuBLAS 할당 실패)."""
    if any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__):
        return True
    text = str(exc)
    return "out of memory" in text.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in text


def oom_message(have_mb, need_mb=NEED_MB):
    have = f"{have_mb / 1024:.0f} GB" if have_mb else "알 수 없음"
    return (
        f"그래픽카드 메모리가 모자랍니다(선화 채색 약 {need_mb / 1024:.0f} GB 필요, 이 PC {have})."
        " 다른 프로그램(게임 · 다른 AI)을 닫고 다시 해 보세요."
    )


def cpu_message():
    return (
        "선화 채색은 NVIDIA 그래픽카드(CUDA)에서만 돕니다. 이 PC 에서 그래픽카드를 찾지 못했거나"
        " 설정이 'CPU 만 쓰기'입니다."
    )


# ── 모델 · 영상 ─────────────────────────────────────────────────────────────


def _import_paths():
    # 상류 코드는 저장소 맨 위를 기준으로 불러온다(models_diffusers · pipelines · cotracker …).
    for p in (HERE, ANIDOC_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 실행 중 인터넷이 막혀 있다 — 허브에 묻지 않게
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


def _vram_mb():
    import torch

    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else 0


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _repo_dir(ctx, repo, probe):
    path = os.path.normpath(ctx.model_path(f"{repo}/{probe}"))
    for _ in probe.split("/"):
        path = os.path.dirname(path)
    return path


def _pipeline(ctx):
    """AniDocPipeline(공식 추론과 같은 부품 · fp16). 한 번 만들면 사이드카가 내려갈 때까지 다시 쓴다."""
    offload = _vram_mb() < FULL_GPU_MB
    if _CACHE["pipe"] is not None and _CACHE["offload"] == offload:
        return _CACHE["pipe"]
    import torch
    from models_diffusers.controlnet_svd import ControlNetSVDModel
    from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from pipelines.AniDoc import AniDocPipeline

    svd_dir = _repo_dir(ctx, SVD, "model_index.json")
    anidoc_dir = _repo_dir(ctx, ANIDOC, "anidoc/unet/config.json")
    ctx.progress(0.02, "load-model")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(anidoc_dir, "anidoc"), subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=True, custom_resume=True
    )
    ctx.progress(0.05, "load-model")
    controlnet = ControlNetSVDModel.from_pretrained(os.path.join(anidoc_dir, "anidoc", "controlnet"), torch_dtype=torch.float16)
    ctx.progress(0.07, "load-model")
    # 공식 코드의 xformers 강제(없으면 멈춤)는 뺐다 — diffusers 기본 어텐션(torch SDPA)으로 돈다.
    pipe = AniDocPipeline.from_pretrained(
        svd_dir, unet=unet, controlnet=controlnet, low_cpu_mem_usage=False, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.set_progress_bar_config(disable=True)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    # diffusers 0.24 는 기본값이 있는 부품(controlnet)을 components 에서 빼서 오프로드 · to() 모두 옮기지 않는다 —
    # 공식 코드처럼 직접 그래픽카드에 둔다(fp16 약 1.4GB).
    controlnet.to("cuda")
    _CACHE.update(pipe=pipe, offload=offload)
    return pipe


def _aux(ctx):
    """선화 추출 · 점 찾기 · 점 맞추기 · 점 따라가기 모델(작다 — 그래픽카드에 둔다)."""
    if _CACHE["aux"] is not None:
        return _CACHE["aux"]
    import torch
    from cotracker.predictor import CoTrackerPredictor
    from lightglue import LightGlue, SuperPoint
    from lineart_extractor.annotator.lineart import LineartDetector

    ctx.progress(0.08, "load-model")
    detector = LineartDetector(DEVICE, ctx.model_path(SK_MODEL))
    extractor = SuperPoint(max_num_keypoints=2000, weights_path=ctx.model_path(SUPERPOINT)).eval().to(DEVICE)
    matcher = LightGlue(features="superpoint", weights_path=ctx.model_path(LIGHTGLUE)).eval().to(DEVICE)
    tracker = CoTrackerPredictor(checkpoint=ctx.model_path(COTRACKER), shift_grid=0)
    tracker.requires_grad_(False)
    tracker.to(DEVICE, dtype=torch.float32)
    _CACHE["aux"] = (detector, extractor, matcher, tracker)
    return _CACHE["aux"]


def _read_video(path):
    """영상 → (RGB PIL 프레임 목록, fps 분수). 화면 회전 정보는 돌려서 보이는 방향으로. 길이 상한을 넘으면 한국어 오류."""
    from fractions import Fraction

    import av
    import numpy as np
    from PIL import Image

    frames = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"'{LABEL_VIDEO}'에 영상이 없습니다. 영상 파일을 이어 주세요.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(24)
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            rot = int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
            if rot:
                arr = np.ascontiguousarray(np.rot90(arr, rot))
            frames.append(Image.fromarray(arr))
            if len(frames) > MAX_FRAMES:
                break
        total = len(frames)
        if total > MAX_FRAMES:
            if stream.frames:
                total = int(stream.frames)
            elif container.duration:
                total = max(total, int(round(container.duration / 1_000_000 * float(rate))))
    if not frames:
        raise ValueError(f"'{LABEL_VIDEO}'에서 프레임을 읽지 못했습니다.")
    problem = check_length(total, float(rate))
    if problem:
        raise ValueError(problem)
    return frames, Fraction(rate).limit_denominator(1001)


def _letterbox(img, box):
    """비율 그대로 흰 여백을 넣어 512×320 으로."""
    from PIL import Image

    w, h, left, top = box
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    canvas.paste(img.convert("RGB").resize((w, h), Image.LANCZOS), (left, top))
    return canvas


def _open_reference(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba)
        im = im.convert("RGB")
    return _letterbox(im, fit_box(*im.size))


def _sketches(frames, source, detector):
    """프레임(512×320 PIL) → 선화 조건 PIL(흰 바탕 검은 선, RGB)."""
    import numpy as np
    from PIL import Image

    out = []
    for img in frames:
        if source == "line":
            sketch = line_to_sketch(np.array(img.convert("L")))
        else:
            line = detector(np.array(img), coarse=False)
            sketch = (line > LINE_THRESHOLD).astype(np.uint8) * 255  # 공식 코드 그대로
        out.append(Image.fromarray(np.repeat(sketch[:, :, np.newaxis], 3, axis=2)))
    return out


def _condition(controlnet_image, ref_image, aux):
    """공식 추론(--all_sketch --matching --tracking)의 ControlNet 조건 만들기를 그대로 옮겼다. (1, 14, 8, h, w) fp16."""
    import numpy as np
    import torch
    import torchvision.transforms as T
    from cotracker.predictor import sample_trajectories_with_ref
    from lightglue.utils import rbd
    from utils import generate_point_map, generate_point_map_frames, safe_round, select_multiple_points

    _detector, extractor, matcher, tracker = aux
    device, num_frames = DEVICE, CHUNK
    controlnet_sketch_condition = [T.ToTensor()(img).unsqueeze(0) for img in controlnet_image]
    controlnet_sketch_condition = torch.cat(controlnet_sketch_condition, dim=0).unsqueeze(0).to(device, dtype=torch.float16)
    controlnet_sketch_condition = (controlnet_sketch_condition - 0.5) / 0.5  # (1,14,3,h,w)
    with torch.no_grad():
        ref_img_value = T.ToTensor()(ref_image).to(device, dtype=torch.float16).to(torch.float32)
        current_img = T.ToTensor()(controlnet_image[0]).to(device, dtype=torch.float16).to(torch.float32)
        feats0 = extractor.extract(ref_img_value)
        feats1 = extractor.extract(current_img)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]
        points0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()
        points1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()
        points0 = safe_round(points0, current_img.shape)
        points1 = safe_round(points1, current_img.shape)
        num_points = min(50, points0.shape[0])
        points0, points1 = select_multiple_points(points0, points1, num_points)
        conditional_pixel_values = ref_img_value.unsqueeze(0).unsqueeze(0)
        conditional_pixel_values = (conditional_pixel_values - 0.5) / 0.5

        video_for_tracker = (controlnet_sketch_condition * 0.5 + 0.5) * 255.0
        queries = np.insert(points1, 0, 0, axis=1)
        queries = torch.from_numpy(queries).to(device, torch.float).unsqueeze(0)
        if queries.shape[1] == 0:
            pred_tracks_sampled = None
        else:
            pred_tracks, pred_visibility = tracker(
                video_for_tracker.to(dtype=torch.float32), queries=queries, grid_size=8, grid_query_frame=0, backward_tracking=False
            )
            pred_tracks_sampled, pred_visibility_sampled, points0_sampled = sample_trajectories_with_ref(
                pred_tracks.cpu(), pred_visibility.cpu(), torch.from_numpy(points0).unsqueeze(0).cpu(),
                max_points=10, motion_threshold=1, vis_threshold=3,
            )
        if pred_tracks_sampled is None:
            mask1 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            mask2 = np.zeros((num_frames, HEIGHT, WIDTH), dtype=np.uint8)
        else:
            pred_tracks_sampled = pred_tracks_sampled.squeeze(0).cpu().numpy()
            pred_visibility_sampled = pred_visibility_sampled.squeeze(0).cpu().numpy()
            points0_sampled = points0_sampled.squeeze(0).cpu().numpy()
            for frame_id in range(num_frames):
                pred_tracks_sampled[frame_id] = safe_round(pred_tracks_sampled[frame_id], current_img.shape)
            points0_sampled = safe_round(points0_sampled, current_img.shape)
            mask1, mask2 = generate_point_map_frames(
                size=current_img.shape, coords0=points0_sampled, coords1=pred_tracks_sampled, visibility=pred_visibility_sampled
            )
        point_map1 = torch.from_numpy(mask1).unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1, 1).to(device, dtype=torch.float16)
        point_map2 = torch.from_numpy(mask2).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.float16)
        point_map = torch.cat([point_map1, point_map2], dim=2)
        conditional_pixel_values_repeat = conditional_pixel_values.repeat(1, num_frames, 1, 1, 1)
        point_map_with_ref = torch.cat([point_map, conditional_pixel_values_repeat], dim=2)
        return torch.cat([controlnet_sketch_condition, point_map_with_ref], dim=2)


def colorize(ctx, inputs, settings):
    t0 = time.monotonic()
    problem = check_inputs(inputs)
    if problem:
        raise ValueError(problem)
    if ctx.device != "cuda":
        raise ValueError(cpu_message())
    source = settings["source"]
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    frames, fps = _read_video(inputs["video"])
    box = fit_box(*frames[0].size)
    frames = [_letterbox(f, box) for f in frames]
    ref_image = _open_reference(inputs["reference"])
    count = len(frames)
    parts = chunk14(frames)
    ctx.log(f"{count} frames @ {float(fps):.3f}fps → {len(parts)} chunk(s), source={source}, seed={seed}")

    _import_paths()
    import numpy as np
    import torch

    results, out = [], None
    try:
        pipe = _pipeline(ctx)
        aux = _aux(ctx)
        ctx.progress(0.1, "work")
        steps_total = len(parts) * STEPS
        done = [0]
        loop_start = time.monotonic()

        def on_step(p, i, t, kwargs):
            ctx.check_cancel()
            done[0] += 1
            ctx.progress(0.1 + 0.85 * done[0] / steps_total, "denoise")
            return kwargs

        for k, part in enumerate(parts):
            ctx.check_cancel()
            random.seed(seed + k)
            np.random.seed((seed + k) % (2**32))
            torch.manual_seed(seed + k)
            sketches = _sketches(part, source, aux[0])
            cond = _condition(sketches, ref_image, aux)
            torch.cuda.empty_cache()  # 점 따라가기가 잡아 둔 캐시를 놓고 생성으로 — 12GB 에서 최고치를 낮춘다
            video = pipe(
                ref_image,
                cond,
                height=HEIGHT,
                width=WIDTH,
                num_frames=CHUNK,
                num_inference_steps=STEPS,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
                generator=torch.manual_seed(seed + k),
                callback_on_step_end=on_step,
            ).frames[0]
            results.append(video)
            del cond
            if k == 0 and len(parts) > 1:
                per = time.monotonic() - loop_start
                estimate = (time.monotonic() - t0) + per * (len(parts) - 1) * 1.1
                ctx.log(f"chunk 1/{len(parts)} in {per:.1f}s, estimate {estimate:.0f}s")
                if estimate > TIME_BUDGET_SEC:
                    raise ValueError(
                        f"이 PC 에서는 약 {math.ceil(estimate / 60)}분이 걸려 한 번에 돌 수 있는 시간(약 {TIME_BUDGET_SEC // 60}분)을"
                        " 넘습니다. 영상을 짧게 잘라 이어 주세요."
                    )
        ctx.progress(0.96, "encode")
        w, h, left, top = box
        colored = [f.crop((left, top, left + w, top + h)) for f in stitch(results, count)]
        out = {"video": ctx.out_path("colorized.mp4")}
        _write_mp4(out["video"], colored, fps)
        if count <= MAX_FRAME_OUTPUTS:
            out["frames"] = []
            for i, f in enumerate(colored):
                path = ctx.out_path(f"frame-{i:03d}.png")
                f.save(path)
                out["frames"].append(path)
        ctx.log(f"done in {time.monotonic() - t0:.1f}s, peak {torch.cuda.max_memory_allocated() // (1024 * 1024)}MB allocated")
    except Exception as exc:  # noqa: BLE001 - 메모리 부족만 쉬운 말로 바꾸고 나머지는 그대로 올린다
        if not is_oom(exc):
            raise
        oom = True
    else:
        oom = False
    finally:
        _free_cuda()
    if oom:
        # except 블록 밖에서 정리하고 새 오류를 낸다 — 블록 안에서 내면 원래 오류(와 그 프레임이 쥔 그래픽카드 텐서)가
        # 새 오류의 __context__ 로 남아 사이드카가 다음 실행까지 메모리를 놓지 못한다.
        _CACHE.update(pipe=None, offload=None, aux=None)
        results = None
        _free_cuda()
        raise RuntimeError(oom_message(_vram_mb()))
    ctx.progress(1.0, "done")
    return out


def _write_mp4(path, frames, fps):
    """PIL RGB 프레임 → H.264 yuv420p mp4(소리 없음). 크기는 fit_box 가 짝수로 맞췄다."""
    import av
    import numpy as np

    w, h = frames[0].size
    container = av.open(path, "w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for img in frames:
            arr = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


NODES = {"colorize": colorize}
