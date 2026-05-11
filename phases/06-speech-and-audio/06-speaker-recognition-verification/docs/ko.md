# 화자 인식 & 검증

> ASR은 "그들은 무엇을 말했는가?"를 묻지만, 화자 인식은 "누가 말했는가?"를 묻습니다. 수학적으로는 임베딩과 코사인 유사도로 동일해 보이지만, 모든 프로덕션 결정은 단일 EER(등오류율) 수치에 달려 있습니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 6 · 02 (스펙트로그램 & 멜), Phase 5 · 22 (임베딩 모델)
**소요 시간:** ~45분

## 문제 정의

사용자가 패스프레이즈를 말할 때, 이 사람이 주장하는 신원과 일치하는지(*검증*, 1:1) 알고 싶을 수 있습니다. 또는 등록 데이터베이스에 있는 첫 번째 사람인지(*식별*, 1:N) 확인하고 싶을 수도 있습니다. 아니면 둘 다 아닌, 알려지지 않은 화자(*오픈셋*)인지 확인하고 싶을 수도 있습니다.

2018년 이전: GMM-UBM + i-벡터. 합리적인 EER(등오류율)을 달성했지만 채널 변화(휴대폰 vs 노트북)와 감정에 취약했습니다. 2018–2022: x-벡터(각도 마진으로 학습된 TDNN 백본). 2022+: ECAPA-TDNN 및 WavLM-large 임베딩. 2026년까지 이 분야는 세 가지 모델과 하나의 지표가 주도하게 됩니다.

그 지표는 **EER**(Equal Error Rate, 등오류율)입니다 — 거짓 수락률 = 거짓 거부률이 되도록 결정 임계값을 설정합니다. 교차점이 EER입니다. 모든 논문, 모든 리더보드, 모든 조달 요청에서 사용됩니다.

## 개념

![임베딩 + 코사인 + EER을 활용한 등록 + 검증 파이프라인](../assets/speaker-verification.svg)

**파이프라인.** 등록: 대상 화자의 5–30초 음성 기록; 고정 차원 임베딩 계산(ECAPA-TDNN의 경우 192-d, WavLM-large의 경우 256-d). 검증: 테스트 발화 임베딩 획득; 코사인 유사도 계산; 임계값과 비교.

**ECAPA-TDNN (2020, 2026년 기준 여전히 지배적).** 채널 어텐션, 전파 및 집계 강조 - 시간 지연 신경망. Squeeze-excitation이 적용된 1D 컨볼루션 블록, 멀티 헤드 어텐션 풀링, 192-d로의 선형 레이어. VoxCeleb 1+2(2,700명 화자, 1.1M 발화)에서 Additive Angular Margin 손실(AAM-softmax)로 학습.

**WavLM-SV (2022+).** 사전 학습된 WavLM-large SSL 백본을 AAM 손실로 파인튜닝. 더 높은 품질이지만 느림 — 300+ MB vs 15 MB.

**x-vector (기준 모델).** TDNN + 통계 풀링. 클래식; CPU/엣지에서 여전히 유용.

**AAM-softmax.** 각도 공간에 마진 `m`을 추가한 표준 소프트맥스: 정답 클래스에 대해 `cos(θ + m)`. 클래스 간 각도 분리 강제. 일반적인 `m=0.2`, 스케일 `s=30`.

## 스코어링

- **코사인.** 등록 및 테스트 임베딩 간 코사인 유사도. 임계값 기반 결정.
- **PLDA (확률적 LDA).** 임베딩을 잠재 공간으로 투영하여 동일 화자 vs 다른 화자에 대한 폐쇄형 우도 비율 계산. 코사인 대비 +10–20% EER 감소. 2020년 이전 표준; 현재는 폐쇄형 설정에서만 사용.
- **스코어 정규화.** `S-norm` 또는 `AS-norm`: 각 스코어를 임포스터 집단의 평균 및 표준편차로 정규화. 도메인 간 평가에 필수.

## 알아두면 좋은 수치 (2026)

| 모델 | VoxCeleb1-O EER | 파라미터 | 처리량 (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (클래식) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 분할 + 임베딩 | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

## 화자 분할(Diarization)

"누가 언제 말했는가"를 다중 화자 클립에서 분석. 파이프라인: VAD → 세그먼트 → 각 세그먼트 임베딩 → 클러스터링(계층적 또는 스펙트럼) → 경계 평활화. 현대 스택: `pyannote.audio` 3.1, 화자 분할 + 임베딩 + 클러스터링을 단일 호출로 통합. 2026년 AMI 기준 SOTA DER은 ~15%(2022년 23%에서 개선).

## 구축 방법

## 단계 1: MFCC 통계 기반 토이 임베딩

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

현저히 최첨단은 아님 — 교육용으로만 사용. `code/main.py`는 합성 화자 데이터에 대한 개념 증명으로 이 방법을 사용합니다.

## 단계 2: 코사인 유사도 + 임계값

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

## 단계 3: 유사도 쌍에서 EER 계산

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

(EER, EER_임계값)을 반환합니다. 둘 다 보고해야 합니다.

## 단계 4: SpeechBrain을 활용한 프로덕션

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 등록: 3-5개의 깨끗한 샘플 임베딩 평균
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 검증
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 일반 임계값; 데이터에 맞게 조정 필요
```

## 단계 5: pyannote로 화자 분할

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| 폐쇄형 1:1 검증, 엣지 | ECAPA-TDNN + 코사인 임계값 |
| 개방형 검증, 클라우드 | WavLM-SV + AS-norm |
| 화자 분할 (회의, 팟캐스트) | `pyannote/speaker-diarization-3.1` |
| 스푸핑 방지 (재생 공격 / 딥페이크 탐지) | AASIST 또는 RawNet2 |
| 초소형 임베디드 (KWS + 등록) | Titanet-Small (NeMo)

## 주의 사항

- **채널 불일치.** VoxCeleb(웹 영상)으로 학습한 모델 ≠ 전화 통화 음성. 항상 대상 채널에서 평가하세요.
- **짧은 발화.** 3초 미만의 테스트 오디오에서 EER(등오류율)이 급격히 저하됩니다.
- **노이즈가 포함된 등록.** 하나의 노이즈가 포함된 등록 샘플이 앵커(anchor)를 오염시킵니다. ≥3개의 깨끗한 샘플을 사용하고 평균을 내세요.
- **조건 간 고정 임계값.** 항상 대상 도메인의 홀드아웃 개발 세트에서 임계값을 조정하세요.
- **정규화되지 않은 임베딩에 코사인 사용.** 먼저 L2-정규화하세요. 그렇지 않으면 크기가 지배적으로 작용합니다.

## Ship It

`outputs/skill-speaker-verifier.md`로 저장. 모델, 등록 프로토콜, 임계값 조정 계획, 사기 방지 장치를 선택.

## 1. 모델 선택
- **기본 모델**: [Speaker Verification 모델명](URL)  
  (예: ECAPA-TDNN, ResNet-LDE, x-vector with PLDA)
- **대체 모델**: [백업 모델명](URL) (예: Wav2Vec 2.0 파인튜닝 버전)

## 2. 등록 프로토콜
1. **음성 샘플 수집**  
   - 사용자당 3~5회 반복 발화 (예: "Hello, my name is [이름]")
   - 배경 소음 최소화를 위한 안내 제공
2. **임베딩 생성**  
   ```python
   # PyTorch 예시 (번역 금지)
   with torch.no_grad():
       embeddings = model(enrollment_audio)
   ```
3. **저장 방식**  
   - 암호화된 벡터 데이터베이스 (AES-256)  
   - 사용자 ID와 임베딩 매핑 테이블 유지

## 3. 임계값 조정 계획
| 단계 | 방법 | 평가 지표 |
|------|------|-----------|
| 1. 초기 설정 | 개발 세트에서 EER(등오류율) 기준 | EER=5% |
| 2. A/B 테스트 | 실제 사용자 1,000명 대상 | FAR=0.1%, FRR=2% |
| 3. 동적 조정 | 월별 오탐률 모니터링 | FAR < 0.5% |

## 4. 사기 방지 장치
- **라이브니스 검증**  
  - 재생 공격 탐지: [Anti-Spoofing 모델명](URL) 통합  
  - 실시간 피치/포먼트 분석
- **다중 인증 계층**  
  ```mermaid
  graph TD
    A[음성 인증] -->|성공| B(2단계 OTP)
    A -->|실패 3회| C[계정 잠금]
    B --> D[접근 승인]
  ```
- **이상 패턴 감지**  
  - 동일 IP에서 10회 이상 시도 시 CAPTCHA 발동  
  - 지리적 위치 불일치 시 추가 확인 절차

> **배포 전 검증**: 실제 환경 테스트 시 [NIST SRE](URL) 벤치마크 준수 확인 필수.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 합성 "화자"(다른 톤 프로파일)를 생성하고, 등록하며, 100개 쌍 시험 목록에서 EER(등오류율)을 계산합니다.
2. **중간.** 30개의 VoxCeleb1 발화(5명 화자 × 각 6개 발화)에 SpeechBrain ECAPA 모델을 사용합니다. 코사인 유사도(cosine similarity) vs PLDA(Probabilistic Linear Discriminant Analysis) 기반 EER을 비교 계산합니다.
3. **어려움.** `pyannote.audio`를 이용해 전체 등록 → 화자 분리(diarization) → 검증(verification) 파이프라인을 구축합니다. AMI 개발 세트에서 DER(Diarization Error Rate)을 평가합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| EER | 헤드라인 메트릭 | False Accept = False Reject인 임계값. |
| 검증(Verification) | 1:1 | "이것이 Alice인가?" |
| 식별(Identification) | 1:N | "화자는 누구인가?" |
| 오픈셋(Open-set) | 미등록 가능 | 테스트 세트에 미등록 화자가 포함될 수 있음. |
| 등록(Enrollment) | 등록 과정 | 화자의 기준 임베딩(embedding) 계산. |
| AAM-softmax | 손실 함수(loss) | 추가 각도 마진(additive angular margin)이 있는 소프트맥스; 클러스터 분리 강제. |
| PLDA | 클래식 스코어링 | 확률적 LDA(Probabilistic LDA); 임베딩 기반 우도비(likelihood-ratio) 스코어링. |
| DER | 다이아리제이션 메트릭 | 다이아리제이션 오류율(Diarization Error Rate) — 누락(miss) + 오경보(false alarm) + 혼동(confusion). |

## 추가 자료

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 고전적인 딥 임베딩 논문.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026년 주요 아키텍처.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 화자 검증(SV) 및 화자 분할(diarization)을 위한 SSL 백본.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 프로덕션 화자 분할 + 임베딩 스택.
- [VoxCeleb 리더보드 (2026년 업데이트)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 모델별 현재 EER 순위.