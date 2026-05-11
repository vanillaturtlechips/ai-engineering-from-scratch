# 평가 — FID, CLIP 점수, 인간 선호도

> 모든 생성 모델 리더보드는 FID, CLIP 점수, 인간 선호도 경기장의 승률을 인용합니다. 각 지표는 실패 모드를 가지고 있으며, 이를 알고 있는 연구자는 이를 악용할 수 있습니다. 실패 모드를 모른다면 실제 개선과 악용 실행을 구분할 수 없습니다.

**유형:** 구축
**언어:** Python
**선수 지식:** 8단계 · 01 (분류 체계), 2단계 · 04 (평가 지표)
**소요 시간:** ~45분

## 문제 정의

생성 모델은 *샘플 품질*과 *조건 충족도*로 평가됩니다. 둘 다 닫힌 형태의 측정 방법이 없습니다. 모델은 10,000개의 이미지를 생성해야 하며, 이 이미지들에 점수를 할당할 수 있는 방법이 필요합니다. 또한 이 점수는 모델 패밀리, 해상도, 아키텍처에 걸쳐 신뢰할 수 있어야 합니다. 2014-2026년 동안 살아남은 세 가지 지표는 다음과 같습니다:

- **FID (Fréchet Inception Distance).** 실제 이미지와 생성된 이미지의 분포를 Inception 네트워크의 특징 공간에서 비교하는 거리 측정법. 값이 낮을수록 좋습니다.
- **CLIP 점수.** 생성된 이미지의 CLIP-이미지 임베딩과 프롬프트의 CLIP-텍스트 임베딩 간의 코사인 유사도. 값이 높을수록 좋으며, 프롬프트 준수도를 측정합니다.
- **인간 선호도.** 동일한 프롬프트에 대해 두 모델을 대결시키고, 인간(또는 GPT-4 수준의 모델)이 더 나은 결과를 선택하도록 한 후 Elo 점수로 집계합니다.

또한 다음과 같은 지표들도 존재합니다: IS (inception score, 대부분 퇴출됨), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. 각각은 이전 지표의 한계를 보완합니다.

## 개념

![FID, CLIP, 선호도: 세 가지 축, 다른 실패 모드](../assets/evaluation.svg)

### FID — 샘플 품질

Heusel et al. (2017). 단계:

1. 실제 이미지 N개와 생성된 이미지 N개에 대해 Inception-v3 특징(2048-D)을 추출.
2. 각 풀에 가우시안을 피팅: 평균 `μ_r, μ_g`와 공분산 `Σ_r, Σ_g` 계산.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

해석: 특징 공간에서 두 다변량 가우시안 간의 프레셰 거리. 값이 낮을수록 분포가 유사함.

실패 모드:
- **작은 N에 편향됨.** FID는 특징 분포에 대한 평균 제곱 오차 — 작은 N은 공분산을 과소추정하여 거짓으로 낮은 FID를 제공. 항상 N ≥ 10,000 사용.
- **Inception-v3 의존성.** Inception-v3는 ImageNet으로 훈련됨. ImageNet과 거리가 먼 도메인(얼굴, 예술, 텍스트 이미지)은 무의미한 FID를 생성. 도메인 특화 특징 추출기 사용.
- **게임화.** Inception 사전 지식에 과적합하면 시각적 품질 개선 없이 낮은 FID를 얻음. CMMD(아래)로 해결.

### CLIP 점수 — 프롬프트 준수

Radford et al. (2021). 생성된 이미지 + 프롬프트에 대해:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30,000개 생성된 이미지에 대해 평균 → 모델 간 비교 가능한 스칼라 값.

실패 모드:
- **CLIP의 한계.** CLIP은 약한 구성적 추론 능력을 가짐("파란 구체 위의 빨간 큐브"는 종종 실패). 모델이 복잡한 프롬프트를 실제로 따르지 않아도 CLIP 점수에서 높은 순위를 얻을 수 있음.
- **짧은 프롬프트 편향.** 짧은 프롬프트는 실제 데이터에서 더 많은 CLIP-이미지 매칭을 가짐. 긴 프롬프트는 기계적으로 낮은 CLIP 점수를 가짐.
- **프롬프트 게임화.** 프롬프트에 "고품질, 4k, 걸작"을 포함하면 이미지-텍스트 결합 개선 없이 CLIP 점수가 부풀려짐.

CMMD(Jayasumana et al., 2024)는 이러한 문제 일부를 해결: Inception 대신 CLIP 특징 사용, 프레셰 거리 대신 최대 평균 불일치 사용. 미묘한 품질 차이 탐지에 더 우수.

### 인간 선호도 — 진실의 기준

프롬프트 풀을 선택. 모델 A와 모델 B로 생성. 인간(또는 강력한 LLM 평가자)에게 쌍으로 제시. 승패를 Elo 또는 Bradley-Terry 점수로 집계. 벤치마크:

- **PartiPrompts (Google)**: 1,600개 다양한 프롬프트, 12개 범주.
- **HPSv2**: 107k 인간 주석, 자동화된 대리 지표로 널리 사용.
- **ImageReward**: 137k 프롬프트-이미지 선호도 쌍, MIT 라이선스.
- **PickScore**: Pick-a-Pic 2.6M 선호도 데이터로 훈련.
- **Chatbot-Arena 스타일 이미지 아레나**: https://imagearena.ai/ 등.

실패 모드:
- **평가자 변동성.** 비전문가와 전문가의 선호도가 다름. 둘 다 사용.
- **프롬프트 분포.** 선별된 프롬프트는 특정 모델에 유리. 항상 문서화.
- **LLM 평가자 보상 해킹.** GPT-4 평가자는 예쁘지만 틀린 출력에 속음. 인간 평가와 삼각측정.

## 함께 사용

프로덕션 평가 보고서에는 다음이 포함되어야 합니다:

1. 10-30k 샘플에 대한 FID(Fréchet Inception Distance) (보유된 실제 분포 대비 샘플 품질).
2. 동일한 샘플과 프롬프트 간의 CLIP 점수 / CMMD(Conditional Maximum Mean Discrepancy) (준수성).
3. 이전 모델 대비 블라인드 아레나에서의 승률 (전체 선호도).
4. 실패 모드 분석: 50개의 무작위 샘플링된 출력, 알려진 문제(손 해부학, 텍스트 렌더링, 일관된 객체 수)에 대해 플래그 지정.

단일 지표는 거짓말입니다. 3개의 상호 보완적 지표 + 정성적 검토가 주장을 뒷받침합니다.

## 구축 방법

`code/main.py`는 합성 "특성 벡터"(Inception 특성 대신 4-D 벡터 사용)에서 FID, CLIP-score 유사, Elo 집계를 구현합니다. 다음을 확인할 수 있습니다:

- 작은 N과 큰 N에서의 FID 계산 — 편향.
- 특성 풀 간의 코사인 유사도로서의 "CLIP 점수".
- 합성 선호도 스트림에서의 Elo 업데이트 규칙.

### 1단계: 4줄로 구현하는 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 2단계: CLIP 스타일 코사인 유사도

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 3단계: Elo 집계

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 함정(Pitfalls)

- **FID at N=1000.** N=10k 미만에서 휴리스틱(heuristic)은 신뢰할 수 없음. 낮은 N의 FID를 보고하는 논문은 결과를 조작(gaming)하고 있음.
- **해상도 간 FID 비교.** Inception의 299×299 리사이즈(resize)는 특징 분포를 변경함. 동일한 해상도에서만 비교해야 함.
- **단일 시드(seed) 보고.** 최소 3개의 시드(seed)를 실행하고 표준편차(std)를 보고해야 함.
- **네거티브 프롬프트(negative prompts)를 통한 CLIP 점수 인플레이션.** 일부 파이프라인은 프롬프트에 과적합(over-fitting)하여 CLIP 점수를 부풀림. 시각적 포화(saturation) 여부를 확인해야 함.
- **프롬프트 중복으로 인한 Elo 편향.** 두 모델이 학습 중에 벤치마크 프롬프트를 모두 접했다면 Elo 점수는 무의미함. 보류된 프롬프트 세트(held-out prompt sets)를 사용해야 함.
- **유료 크라우드소싱 기반 인간 평가의 편향.** Prolific, MTurk 평가자는 젊은 층 / 기술 친화적 집단으로 편향됨. 예술/디자인 분야 전문가를 혼합하여 평가해야 함.

## 사용 방법

2026년 프로덕션 평가 프로토콜:

| 기둥 | 최소 기준 | 권장 기준 |
|--------|---------|-------------|
| 샘플 품질 | 10k 샘플에 대한 FID(Fréchet Inception Distance) vs. 보유 중인 실제 데이터 | + 5k 샘플에 대한 CMMD(Central Multidimensional Scaling Distance) + 카테고리별 부분 집합 FID |
| 프롬프트 준수 | 30k 샘플에 대한 CLIP 점수 | + HPSv2 + ImageReward + VQA(Vision Question Answering) 스타일 질문 답변 |
| 선호도 | 200개의 블라인드 쌍 비교 vs. 기준 모델 | + 2000개의 쌍 비교 인간 평가 + LLM 평가자 + Chatbot Arena |
| 실패 분석 | 50개의 수동 플래그 지정 | 500개의 수동 플래그 지정 + 자동화된 안전 분류기 |

네 가지 기둥을 모두 포함한 보고서 = 공식 주장. 단일 기둥만 사용 = 마케팅.

## Ship It

`outputs/skill-eval-report.md`를 저장하세요. Skill은 새로운 모델 체크포인트 + 베이스라인을 입력으로 받아 샘플 크기, 메트릭, 실패 모드 프로브, 승인 기준 등이 포함된 전체 평가 계획을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 동일한 합성 분포에서 N=100과 N=1000일 때 FID를 비교합니다. 편향 크기를 보고합니다.
2. **중간.** 합성 CLIP 스타일 특징에서 CMMD를 구현합니다(Jayasumana et al., 2024의 공식 참조). 품질 차이에 대한 민감도를 FID와 비교합니다.
3. **어려움.** HPSv2 설정을 재현합니다: Pick-a-Pic의 하위 집합에서 1000개의 이미지-프롬프트 쌍을 가져와, 선호도에 따라 작은 CLIP 기반 평가기를 파인튜닝(fine-tuning)하고, 홀드아웃 세트와의 일치도를 측정합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| FID | "Fréchet Inception Distance" | 실제 vs 생성 이미지 Inception 특징의 가우시안 분포 간 Fréchet 거리 |
| CLIP 점수 | "텍스트-이미지 유사도" | CLIP 이미지 및 텍스트 임베딩 간 코사인 유사도 |
| CMMD | "FID의 대체 지표" | CLIP 특징 MMD; 편향 감소, 가우시안 가정 불필요 |
| IS | "Inception 점수" | Exp KL(p(y|x) || p(y)); 현대 모델에서 낮은 상관관계, 사용 중단 |
| HPSv2 / ImageReward / PickScore | "학습된 선호도 프록시" | 인간 선호도 데이터로 학습된 소형 모델; 자동 평가자 역할 |
| Elo | "체스 레이팅" | Bradley-Terry 방식의 쌍대비교 승률 집계 |
| PartiPrompts | "벤치마크 프롬프트 세트" | 12개 카테고리에 걸친 Google 선정 1,600개 프롬프트 |
| FD-DINO | "자기지도 대체 지표" | DINOv2 특징 기반 FD; ImageNet 외 도메인에서 더 우수 |

## 프로덕션 노트: 평가는 추론 작업이기도 합니다

10k 샘플에 FID를 실행하는 것은 10k 개의 이미지를 생성하는 것을 의미합니다. 단일 L4에서 50단계 SDXL 베이스로 1024² 해상도를 처리할 경우, 이는 단일 요청 추론 시 약 11시간이 소요됩니다. 평가 예산은 실제로 존재하며, 이는 정확히 오프라인 추론 시나리오와 동일한 프레임입니다(처리량 최대화, TTFT 무시):

- **배치 우선, 지연 시간은 무시.** 오프라인 평가 = 메모리에 맞는 최대 크기의 정적 배치 처리. 80GB H100에서 `num_images_per_prompt=8`로 `pipe(...).images`를 실행하면 단일 요청보다 4-6배 빠른 벽시계 시간을 기록합니다.
- **실제 특징 캐싱.** 실제 참조 세트에 대한 Inception(FID) 또는 CLIP(CLIP 점수, CMMD) 특징 추출은 *한 번만* 실행되고 `.npz`로 저장됩니다. 평가마다 재계산하지 마세요.

CI / 회귀 게이트의 경우: PR당 500개 샘플 하위 집합에서 FID + CLIP 점수 실행(~30분); 매일 밤 전체 10k FID + HPSv2 + Elo 실행.

## 추가 자료

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 논문.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 실패 모드 조사.