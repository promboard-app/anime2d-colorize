<!-- version: 1.0.0 | lastUpdated: 2026-10-31 -->
# NOTICE — 담아 온 코드 · 받는 모델 · 고친 곳

이 플러그인의 PromBoard 코드(`nodes/main.py` · `tests/`)는 Apache-2.0 이다(`LICENSE`). 아래 상류 코드를 함께 담았고, 각 폴더의 라이선스를 따른다.

## 담아 온 코드

| 폴더 | 상류 | 라이선스 |
|---|---|---|
| `nodes/anidoc/` (models_diffusers · pipelines · utils.py) | [yihao-meng/AniDoc](https://github.com/yihao-meng/AniDoc) `77e0696` | Apache-2.0 (`nodes/anidoc/LICENSE`) |
| `nodes/anidoc/lineart_extractor/annotator/lineart/` | AniDoc 사본 ← [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings) | MIT (같은 폴더 `LICENSE`) |
| `nodes/cotracker/` (`build/` · 학습 스크립트 폴더 `project/` 제외) | AniDoc 사본(점 고르기 함수 추가본) ← [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | CC-BY-NC-4.0 (`nodes/cotracker/project/LICENSE.md`) |
| `nodes/lightglue/` (superpoint.py 밖) | AniDoc 사본 ← [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 (`nodes/lightglue/LICENSE`) |
| `nodes/lightglue/superpoint.py` | 위와 같음 ← Magic Leap SuperPoint | [SuperPoint 라이선스(비상업 연구용)](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) — `LicenseRef-SuperPoint-NC` |

## 받는 모델 (실행 준비가 원래 출처에서 받는다)

| 파일 | 출처 | 라이선스 |
|---|---|---|
| AniDoc unet · controlnet | HF `Yhmeng1106/anidoc` | 저장소 표기 MIT. Stable Video Diffusion 미세조정본이라 Stability AI Community License 도 따른다 |
| SVD img2vid-xt image_encoder · vae (fp16) · 설정 | HF `stabilityai/stable-video-diffusion-img2vid-xt` | [Stability AI Community License](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/9e43909513c6714f1bc78bcb44d96e733cd242aa/LICENSE.md) |
| `sk_model.pth` (선화 뽑기) | HF `lllyasviel/Annotators` | informative-drawings 가중치(MIT) |
| `cotracker2.pth` | HF `facebook/cotracker` | CC-BY-NC-4.0 |
| `superpoint_v1.pth` | GitHub `cvg/LightGlue` 릴리스 `v0.1_arxiv` | SuperPoint 라이선스(비상업 연구용) |
| `superpoint_lightglue.pth` | 위와 같음 | Apache-2.0 |

This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved. Powered by Stability AI. 이 플러그인은 Stability AI 모델 파일을 고치거나 다시 배포하지 않고, 원래 저장소에서 받아 그대로 읽는다.

## 고친 곳 (코드에 `# PromBoard:` 주석)

- `nodes/anidoc/lineart_extractor/annotator/lineart/__init__.py` — basicsr 로 실행 중에 받던 가중치를 미리 받은 경로로 읽는다. `torch.load(weights_only=True, mmap=True)`. AniDoc 추론이 쓰지 않는 `sk_model2.pth` 는 받지 않는다(주면 읽는다).
- `nodes/anidoc/models_diffusers/unet_spatio_temporal_condition.py` — 오류 문구의 패키지 설치 안내 한 줄을 뺐다(PromBoard 레지스트리 검사가 실행 중 설치로 읽는다).
- `nodes/anidoc/utils.py` — scikit-image · moviepy 를 쓰는 함수 안에서 불러온다(moviepy 2.x 꼴 `from moviepy import ImageSequenceClip`). PromBoard 노드는 그 함수를 부르지 않아 두 패키지를 설치하지 않는다.
- `nodes/cotracker/models/build_cotracker.py` — `torch.load(경로, weights_only=True, mmap=True)`.
- `nodes/lightglue/superpoint.py` · `lightglue.py` — `torch.hub` 로 받던 가중치를 설정 `weights_path` 의 경로에서 `torch.load(weights_only=True, mmap=True)` 로 읽는다.
- 쓰지 않는 AniDoc 파일(`models_diffusers/` 의 adapter_model · mutual_self_attention · refUnet · unet_..._interp, `scripts_infer/`, `datasets/`, `lineart_extractor` 의 canny · hed · lineart_anime)과 LightGlue 예제 · 그림은 담지 않았다.
- 추론 스크립트(`scripts_infer/anidoc_inference.py`)는 담지 않고, 그 조건 만들기(`--all_sketch --matching --tracking --max_point 10`)를 `nodes/main.py` 의 `_condition` 으로 옮겼다. xformers 가 없으면 멈추던 곳(117–123행)은 옮기지 않아 diffusers 기본 어텐션(torch SDPA)으로 돈다.
