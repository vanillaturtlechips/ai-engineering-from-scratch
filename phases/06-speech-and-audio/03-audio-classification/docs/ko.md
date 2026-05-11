# 오디오 분류 — MFCC 기반 k-NN에서 AST 및 BEATs까지

> "개 짖음 vs 사이렌"부터 "이 음성은 어떤 언어인가"까지 모두 오디오 분류 문제입니다. 특징은 멜(mels)입니다. 아키텍처는 매 10년마다 변화합니다. 평가는 AUC, F1, 클래스별 재현율(recall)로 유지됩니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** 6단계 · 02 (스펙트로그램 & 멜), 3단계 · 06 (CNNs), 5단계 · 08 (텍스트용 CNNs & RNNs)
**소요 시간:** ~75분

## 문제 정의

10초 분량의 클립을 받습니다. "이것은 무엇인가?"를 알고 싶습니다: 도시 소리(사이렌, 드릴, 개), 음성 명령(예/아니오/정지), 언어 식별(en/es/ar), 화자 감정(분노/중립), 또는 환경 소리(실내/실외, 웅성거림). 이 모든 것은 *오디오 분류* 작업이며, 2026년에는 기준 아키텍처가 성숙해졌습니다: 로그-멜(log-mel) → CNN 또는 트랜스포머(Transformer) → 소프트맥스(softmax).

핵심 어려움은 네트워크가 아닙니다. 데이터입니다. 오디오 데이터셋은 극심한 클래스 불균형, 강한 도메인 변화(clean vs noisy), 레이블 노이즈("도시 웅성거림" vs "레스토랑 소음"을 누가 결정했는지?) 문제를 안고 있습니다. 문제의 80%는 큐레이션, 증강, 평가에 있으며, CNN을 트랜스포머로 교체하는 것이 아닙니다.

## 개념

![오디오 분류 계층: MFCC 기반 k-NN에서 AST, BEATs까지](../assets/audio-classification.svg)

**MFCC 기반 k-NN (1990년대 기준선).** 클립별 MFCC를 평탄화하고, 레이블이 지정된 뱅크와 코사인 유사도를 계산한 후 상위 K개의 다수결 결과를 반환합니다. 깨끗하고 작은 데이터셋(Speech Commands, ESC-50)에서 놀랍도록 강력합니다. GPU 없이 실행됩니다.

**로그-멜에 대한 2D CNN (2015-2019).** `(T, n_mels)` 로그-멜을 이미지로 처리합니다. ResNet-18 또는 VGG 스타일을 적용합니다. 시간 축을 글로벌 평균 풀링합니다. 클래스별 소프트맥스를 수행합니다. 2026년 대부분의 캐글 경연에서 여전히 기준선입니다.

**오디오 스펙트로그램 트랜스포머, AST (2021-2024).** 로그-멜을 패치로 분할(예: 16×16 패치)하고 위치 임베딩을 추가한 후 ViT에 입력합니다. 감독 학습에서 AudioSet(mAP 0.485)의 최신 기술입니다.

**BEATs 및 WavLM-base (2024-2026).** 수백만 시간 분량의 자기 지도 사전 학습. 감독 데이터의 1-10%로 작업을 미세 조정합니다. 2026년에는 비음성 오디오의 기본 시작점입니다. BEATs-iter3는 AudioSet에서 AST보다 1-2 mAP 높은 성능을 보이며 1/4의 계산량만 사용합니다.

**동결된 백본으로서의 Whisper-인코더 (2024).** Whisper의 인코더를 가져와 디코더를 제거하고 선형 분류기를 연결합니다. 오디오 증강 없이도 언어 ID 및 단순 이벤트 분류에서 근-SOTA 성능을 달성합니다. "무료 점심" 기준선입니다.

## 클래스 불균형이 진짜 도전 과제

ESC-50: 50개 클래스, 각 클래스당 40개 클립 — 균형 잡혀 있고 쉽습니다. UrbanSound8K: 10개 클래스, 10:1 불균형. AudioSet: 632개 클래스로 100,000:1의 긴 꼬리. 효과적인 기법:

- 훈련 중 균형 샘플링(평가 시에는 사용하지 않음).
- Mixup: 두 클립(및 레이블)을 선형 보간하여 증강.
- SpecAugment: 무작위 시간 및 주파수 대역을 마스킹. 간단하지만 중요합니다.

## 평가

- 다중 클래스 배타적(Speech Commands): 상위 1개 정확도, 상위 5개 정확도.
- 다중 클래스 다중 레이블(AudioSet, UrbanSound 스타일): 평균 평균 정밀도(mAP).
- 심하게 불균형한 경우: 클래스별 재현율 + 매크로 F1.

2026년 기준 알아야 할 수치:

| 벤치마크 | 기준선 | 2026 SOTA | 출처 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 논문 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 리더보드 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 결과 |

## 구축 방법

## 1단계: 특징 추출

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

## 2단계: 고정 길이 요약

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

간단하지만 강력한 방법: 시간 축 평균 + 분산은 13계수 MFCC에 대해 26차원 고정 임베딩을 제공합니다. 즉시 실행되며, 2017년 기준으로 ESC-50에서 최첨단 NN 기준선을 능가합니다.

## 3단계: k-최근접 이웃

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

## 4단계: 로그-멜 기반 CNN으로 업그레이드

PyTorch에서:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300만 개의 파라미터. 단일 RTX 4090에서 ESC-50에 대해 약 10분 동안 학습. 80%+ 정확도.

## 5단계: 2026년 기본 설정 — BEATs 파인튜닝

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs의 경우 `microsoft/BEATs-base`를 `beats` 라이브러리를 통해 사용. transformers API는 동일한 구조를 가집니다.

## 사용 방법

2026 스택:

| 상황 | 시작 방법 |
|-----------|-----------|
| 작은 데이터셋 (<1000 클립) | MFCC 평균에 대한 k-NN (기준선) + 오디오 증강 |
| 중간 데이터셋 (1K–100K) | BEATs 또는 AST 파인튜닝 |
| 큰 데이터셋 (>100K) | 처음부터 학습 또는 Whisper-인코더 파인튜닝 |
| 실시간, 엣지 | 40-MFCC CNN, int8 양자화 (KWS 스타일) |
| 다중 레이블 (AudioSet) | BCE 손실 + 믹스업 + SpecAugment를 사용한 BEATs-iter3 |
| 언어 식별 | MMS-LID, SpeechBrain VoxLingua107 기준선 |

결정 규칙: **새로운 모델이 아닌 고정된 백본(backbone)으로 시작하세요**. BEATs 헤드를 파인튜닝하면 몇 주가 아닌 몇 시간 내에 SOTA의 95%를 달성할 수 있습니다.

## Ship It

`outputs/skill-classifier-designer.md`로 저장. 주어진 오디오 분류 작업을 위해 아키텍처, 증강 기법, 클래스 균형 전략, 평가 지표를 선택하세요.

## 1. 아키텍처 선택
- **CNN + Transformer Hybrid**  
  - 오디오 스펙트로그램 처리에 적합한 **CNN(Convolutional Neural Network)**으로 지역 패턴 추출  
  - **Transformer** 레이어로 장거리 의존성 모델링  
  - 예: `AST(Audio Spectrogram Transformer)` 또는 `Wav2Vec 2.0` 기반 파인튜닝

## 2. 데이터 증강 기법
- **시간/주파수 영역 증강**  
  - 시간 영역: `피치 시프트(pitch shift)`, `시간 스트레칭(time stretch)`  
  - 주파수 영역: `멜-스펙트로그램(mel-spectrogram) 마스킹`, `노이즈 주입(noise injection)`  
  - 확률적 증강: `SpecAugment` 적용

## 3. 클래스 균형 전략
- **가중치 조정 + 리샘플링**  
  - **클래스 가중치(class weight)**를 손실 함수(loss function)에 적용  
  - 소수 클래스 오버샘플링: `SMOTE(Synthetic Minority Over-sampling Technique)` 또는 `Random Over/Under-sampling`  
  - **Focal Loss**로 어려운 샘플에 집중

## 4. 평가 지표
- **주요 지표**  
  - **가중치 없는 평균 정밀도/재현율(Unweighted Average Precision/Recall, UAP/UAR)**  
  - **F1-Score(Macro)**로 클래스 불균형 고려  
  - **혼동 행렬(Confusion Matrix)**로 클래스별 성능 분석  
- **보조 지표**  
  - 정확도(Accuracy) - 데이터 균형 시 참고  
  - AUC-ROC - 이진 분류 시 활용  

## 5. 구현 예시 (PyTorch)
```python
# 모델 정의
model = ASTModel.from_pretrained("facebook/ast-10k-16k").finetune(num_labels=10)

# 손실 함수 (클래스 가중치 적용)
class_weights = torch.tensor([1.0, 2.5, 3.0, ...])  # 클래스별 가중치
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 증강 파이프라인 (torchaudio)
transform = transforms.Compose([
    transforms.PitchShift(n_steps=2),
    transforms.TimeStretch(factor=1.2),
    transforms.MelSpectrogram(sample_rate=16000)
])
```

## 6. 검증 전략
- **Stratified K-Fold Cross-Validation** (K=5)  
- 테스트 세트: 실제 환경 데이터 포함 (배경 노이즈, 다양한 화자)  

> **참고**: 하드웨어 제약 시 `EfficientNet` 기반 오디오 모델(`AudioSpectrogramModel`)로 경량화 가능.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 스크립트는 4-클래스 합성 데이터셋(다양한 피치의 순수 톤)에서 k-NN MFCC 베이스라인을 훈련합니다. 혼동 행렬(confusion matrix)을 보고하세요.
2. **중간.** `summarize`를 [평균(mean), 분산(var), 왜도(skew), 첨도(kurtosis)]로 대체하세요. 동일한 합성 데이터셋에서 4-모멘트 풀링(4-moment pooling)이 평균+분산(mean+var)보다 성능이 좋은지 확인하세요.
3. **어려움.** `torchaudio`를 사용하여 ESC-50 폴드 1에 2D CNN을 훈련하세요. 5-폴드 교차 검증 정확도(5-fold cross-validation accuracy)를 보고하세요. SpecAugment(시간 마스크 = 20, 주파수 마스크 = 10)를 추가하고 변화량(delta)을 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| AudioSet | 오디오의 ImageNet | Google의 200만 개 클립, 632개 클래스 약감독(weakly-labeled) YouTube 데이터셋. |
| ESC-50 | 소규모 분류 벤치마크 | 50개 클래스 × 40개 클립의 환경 음향. |
| AST | 오디오 스펙트로그램 트랜스포머 | 로그-멜(log-mel) 패치에 적용된 ViT; 2021 SOTA(State Of The Art). |
| BEATs | 자기지도 학습 오디오 | Microsoft 모델, 2026년 기준 iter3가 AudioSet에서 성능 선두. |
| Mixup | 페어 증강 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | 마스크 기반 증강 | 스펙트로그램의 무작위 시간 및 주파수 대역을 0으로 설정. |
| mAP | 주요 다중 레이블 평가 지표 | 클래스와 임계값(threshold) 간 평균 정밀도(mean average precision). |

## 추가 자료

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024년 기록 보유 아키텍처.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 기본 모델.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 주요 오디오 증강 기법.
- [Piczak (2015). ESC-50 데이터셋](https://github.com/karolpiczak/ESC-50) — 지속적으로 사용되는 50개 클래스 벤치마크.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632개 클래스 YouTube 분류 체계; 여전히 표준(gold standard)으로 인정됨.