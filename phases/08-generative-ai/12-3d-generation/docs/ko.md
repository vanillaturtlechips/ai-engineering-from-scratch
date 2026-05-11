# 3D 생성

> 3D는 2D-to-3D 활용이 가장 강력한 모달리티입니다. 2023년의 획기적인 발전은 3D 가우시안 스플래팅(3D Gaussian Splatting)이었습니다. 2024-2026년 생성형 AI 추진은 멀티뷰 디퓨전(multi-view diffusion) + 3D 재구성을 계층적으로 적용하여 단일 프롬프트나 사진으로부터 객체와 장면을 생성합니다.

**유형:** 학습
**언어:** Python
**선수 지식:** 4단계(비전), 8단계 · 07(잠재 디퓨전)
**소요 시간:** ~45분

## 문제

3D 콘텐츠는 고통스럽습니다:

- **표현 방식.** 메시(mesh), 포인트 클라우드(point cloud), 복셀 그리드(voxel grid), 부호 있는 거리 필드(signed distance field, SDF), 뉴럴 래디언스 필드(neural radiance field, NeRF), 3D 가우시안(3D Gaussian). 각각 장단점이 있습니다.
- **데이터 부족.** ImageNet에는 1400만 개의 이미지가 있습니다. 가장 큰 정제된 3D 데이터셋(Objaverse-XL, 2023)에는 약 1000만 개의 객체가 있지만 대부분 품질이 낮습니다.
- **메모리.** 512³ 복셀 그리드는 1억 2800만 개의 복셀로 구성됩니다. 유용한 장면 NeRF는 광선당 100만 개의 샘플이 필요합니다. 생성은 재구성보다 더 어렵습니다.
- **지도 학습.** 2D 이미지의 경우 픽셀이 있습니다. 3D의 경우 일반적으로 소수의 2D 뷰만 있고 이를 3D로 변환해야 합니다.

2026년 기술 스택은 두 문제를 분리합니다. 첫째, 확산 모델(diffusion model)로 *2D 다중 뷰 이미지*를 생성합니다. 둘째, 해당 이미지에 *3D 표현 방식*(보통 가우시안 스플래팅)을 피팅합니다.

## 개념

![3D 생성: 다중 뷰 확산 + 3D 재구성](../assets/3d-generation.svg)

### 표현: 3D 가우시안 스플래팅 (Kerbl et al., 2023)

장면을 약 100만 개의 3D 가우시안 구름으로 표현합니다. 각각은 59개의 매개변수를 가집니다: 위치(3), 공분산(6, 또는 쿼터니언 4 + 스케일 3), 불투명도(1), 구면 조화 색상(3차수 48, 0차수 3).

렌더링 = 투영 + 알파 합성. 빠름(4090에서 1080p 기준 약 100fps). 미분 가능. 실제 사진을 기준으로 경사 하강법으로 적합화. 소비자 GPU에서 5-30분 내 장면 적합화 가능.

2023-2024년 두 가지 혁신:
- **생성형 가우시안 스플랫.** LGM, LRM, InstantMesh 같은 모델은 하나 또는 소수의 이미지로부터 직접 가우시안 구름을 예측합니다.
- **4D 가우시안 스플래팅.** 동적 장면을 위한 프레임별 오프셋이 있는 가우시안.

### 다중 뷰 확산

사전 훈련된 이미지 확산 모델을 미세 조정하여 텍스트 프롬프트나 단일 이미지로부터 동일한 객체의 여러 일관된 뷰를 생성합니다. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). 일반적으로 객체 주변 4-16개의 뷰를 출력하며, 가우시안 스플래팅이나 NeRF를 통해 3D로 변환됩니다.

### 텍스트-3D 파이프라인

| 모델 | 입력 | 출력 | 시간 |
|-------|-------|--------|------|
| DreamFusion (2022) | 텍스트 | SDS를 통한 NeRF | 자산당 약 1시간 |
| Magic3D | 텍스트 | 메시 + 텍스처 | 약 40분 |
| Shap-E (OpenAI, 2023) | 텍스트 | 암시적 3D | 약 1분 |
| SJC / ProlificDreamer | 텍스트 | NeRF / 메시 | 약 30분 |
| LRM (Meta, 2023) | 이미지 | 트라이플레인 | 약 5초 |
| InstantMesh (2024) | 이미지 | 메시 | 약 10초 |
| SV3D (Stability, 2024) | 이미지 | 새로운 뷰 | 약 2분 |
| CAT3D (Google, 2024) | 1-64 이미지 | 3D NeRF | 약 1분 |
| TripoSR (2024) | 이미지 | 메시 | 약 1초 |
| Meshy 4 (2025) | 텍스트 + 이미지 | PBR 메시 | 약 30초 |
| Rodin Gen-1.5 (2025) | 텍스트 + 이미지 | PBR 메시 | 약 60초 |
| Tencent Hunyuan3D 2.0 (2025) | 이미지 | 메시 | 약 30초 |

2025-2026년 방향: 게임 엔진에 적합한 PBR 재질을 가진 직접 텍스트-메시 모델. 다중 뷰 확산 중간 단계는 여전히 일반 객체에 대해 가장 성능이 좋은 레시피입니다.

### NeRF (참고용)

신경 방사장 필드 (Mildenhall et al., 2020). 작은 MLP가 `(x, y, z, 시야 방향)`을 입력받아 `(색상, 밀도)`를 출력합니다. 광선을 따라 적분하여 렌더링. 메시 기반 새로운 뷰 합성보다 품질은 우수하지만 렌더링 속도는 100-1000배 느림. 대부분의 실시간 사용 사례에서는 가우시안 스플래팅에 의해 대체되었지만 연구 분야에서는 여전히 주류.

## 빌드하기

`code/main.py`는 장난감 2D "가우시안 스플래팅" 피팅을 구현합니다: 합성 대상 이미지(부드러운 그라데이션)를 2D 가우시안 스플랫의 합으로 표현합니다. 대상 이미지와 일치하도록 기울기 하강법을 사용해 위치, 색상, 공분산을 최적화합니다. 두 가지 핵심 연산인 순방향 렌더링(스플랫 + 알파 합성)과 기울기 하강법을 통한 피팅을 확인할 수 있습니다.

### 1단계: 2D 가우시안 스플랫

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 2단계: 스플랫 합산을 통한 렌더링

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

실제 3D 가우시안 스플래팅은 깊이 순으로 가우시안을 정렬하고 알파 합성합니다. 이 2D 장난감은 단순히 합산만 수행합니다.

### 3단계: 기울기 하강법을 통한 피팅

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 함정(Pitfalls)

- **뷰 불일치(View inconsistency).** 4개의 뷰를 독립적으로 생성하고 객체 구조에 대해 불일치할 경우 3D 피팅이 흐릿해집니다. 해결 방법: 공유 어텐션(shared attention)을 사용한 멀티뷰 확산(multi-view diffusion).
- **뒷면 환각(Back-side hallucination).** 단일 이미지 → 3D는 보이지 않는 면을 생성해야 합니다. 품질이 크게 달라집니다.
- **가우시안 스플랫 폭발(Gaussian splat explosion).** 제약 없는 훈련은 10M 스플랫까지 성장시키고 과적합(overfitting)을 유발합니다. 밀도 증가(densification) + 가지치기(pruning) 휴리스틱(3D-GS 원본 논문 참조)이 필수적입니다.
- **위상 문제(Topology issues).** 암시적 필드(SDFs)에서 생성된 메시는 종종 구멍이나 자기 교차를 포함합니다. 배포 전 리메셔(예: 블렌더의 복셀 리메셔)를 실행하세요.
- **훈련 데이터 라이선스(License of training data).** Objaverse는 라이선스가 혼합되어 있으며, 상업적 사용은 모델마다 다릅니다.

## 사용 방법

| 작업 | 2026년 추천 도구 |
|------|------------------|
| 사진 기반 장면 재구성 | 가우시안 스플래팅(3DGS, Gsplat, Scaniverse) |
| 게임용 텍스트-3D 객체 생성 | Meshy 4 또는 Rodin Gen-1.5 (PBR 출력) |
| 이미지-3D 변환 | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 소수 이미지 기반 신뷰 합성 | CAT3D, SV3D |
| 동적 장면 재구성 | 4D 가우시안 스플래팅 |
| 아바타 / 옷 입은 인간 | 가우시안 아바타, HUGS |
| 연구 / 최신 기술(SOTA) | 지난주에 출시된 최신 도구 |

게임 또는 전자상거래 파이프라인에서 프로덕션 3D를 구현할 경우: Meshy 4 또는 Rodin Gen-1.5에서 출력된 PBR 메시를 Unity/Unreal에 바로 적용할 수 있습니다.

## Ship It

`outputs/skill-3d-pipeline.md`를 저장하세요. Skill은 3D 요구사항(입력: 텍스트 / 단일 이미지 / 여러 이미지; 출력: 메시 / 스플랫 / NeRF; 사용 사례: 렌더링 / 게임 / VR)을 받아 다음과 같은 출력을 생성합니다: 파이프라인(다중 뷰 확산 + 피팅, 또는 직접 메시 모델), 기본 모델, 반복 예산, 토폴로지 후처리, 필요한 재질 채널.

## 연습 문제

1. **쉬움.** `code/main.py`를 4, 16, 64개의 가우시안(Gaussian)으로 실행해 보세요. 목표(target) 대비 최종 MSE(평균 제곱 오차)를 보고하세요.
2. **중간.** 컬러 가우시안(RGB)으로 확장하세요. 재구성 결과가 목표 색상 패턴과 일치하는지 확인하세요.
3. **어려움.** gsplat 또는 Nerfstudio를 사용하여 50장의 사진으로 촬영된 실제 객체를 재구성하세요. 피팅 시간과 보류된 뷰(held-out views)에서의 최종 SSIM(구조적 유사성 지수)을 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|---------------------|-----------|
| 3D 가우시안 스플래팅(3D Gaussian Splatting) | "3DGS" | 3D 가우시안 구름 형태의 장면; 미분 가능한 알파 합성 렌더링. |
| NeRF | "Neural radiance field" | 3D 포인트에서 색상 + 밀도를 출력하는 MLP; 광선 적분을 통한 렌더링. |
| 트라이플레인(Triplane) | "Three 2-D planes" | 3D를 세 개의 2D 축 정렬 피처 그리드로 분해; 볼륨 방식보다 효율적. |
| SDS | "Score distillation sampling" | 2D 확산 모델의 스코어를 의사 그래디언트로 사용하여 3D 모델 학습. |
| 멀티뷰 확산(Multi-view diffusion) | "Many views at once" | 일관된 카메라 뷰 배치를 출력하는 확산 모델. |
| PBR | "Physically-based rendering" | 알베도, 거칠기, 금속성, 노멀 채널을 가진 재질. |
| 덴시피케이션(Densification) | "Grow splats" | 3DGS 학습 휴리스틱: 고 그래디언트 영역에서 스플랫 분할/복제. |

## 프로덕션 노트: 3D는 아직 공유 기반이 없음

이미지(잠재 확산 + DiT) 및 비디오(시공간 DiT)와 달리, 3D는 2026년 기준으로 단일 지배적인 런타임이 없습니다. 프로덕션 결정 트리는 표현 방식에 따라 분기됩니다:

- **NeRF / 트라이플레인.** 추론은 레이 마칭(ray-marching) + 샘플당 MLP 순전파(forward)입니다. 512² 렌더링에는 수백만 번의 MLP 순전파가 필요합니다. 레이 샘플을 공격적으로 배치 처리하세요. SDPA/xformers가 적용됩니다.
- **다관점 확산 + LRM 재구성.** 2단계 파이프라인입니다. 1단계(다관점 DiT)는 레슨 07과 동일한 확산 서버입니다. 2단계(LRM 트랜스포머)는 뷰(view) 전체에 대한 단일 순전파(one-shot forward pass)입니다. 전체 지연 시간 프로파일은 "확산 + 단일 순전파"입니다. 각 단계별 서빙 프리미티브를 적절히 선택하세요.
- **SDS / DreamFusion.** 추론이 아닌 자산별 최적화입니다. 요청 핸들러가 아닌 작업(job)을 구축하세요.

대부분의 2026년 제품에서 올바른 접근 방식은 "요청 시 다관점 확산 모델을 실행하고, 비동기적으로 3DGS로 재구성한 후, 실시간 뷰잉을 위해 3DGS를 서빙하는 것"입니다. 이는 GPU 추론 서버(빠름)와 오프라인 최적화기(느림) 간에 작업 부하를 명확하게 분리합니다.

## 추가 자료

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 멀티뷰 디퓨전(multi-view diffusion).
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.