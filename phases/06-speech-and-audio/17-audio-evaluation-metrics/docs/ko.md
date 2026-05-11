# 오디오 평가 — WER, MOS, UTMOS, MMAU, FAD, 그리고 오픈 리더보드

> 측정할 수 없는 것은 출시할 수 없다. 이 강의에서는 모든 오디오 작업에 대한 2026년 메트릭을 소개합니다: ASR(WER, CER, RTFx), TTS(MOS, UTMOS, SECS, WER-on-ASR-round-trip), 오디오-언어(MMAU, LongAudioBench), 음악(FAD, CLAP), 화자(EER). 비교를 위한 리더보드도 함께 다룹니다.

**유형:** 학습
**언어:** Python
**선수 지식:** 6단계 · 04, 06, 07, 09, 10; 2단계 · 09 (모델 평가)
**소요 시간:** ~60분

## 문제

모든 오디오 작업에는 여러 메트릭이 있으며, 각각 다른 축을 측정합니다. 잘못된 메트릭을 사용하면 대시보드에서는 훌륭해 보이지만 실제 운영에서는 형편없는 모델을 출시하게 됩니다. 2026년 표준 목록:

| 작업 | 주요 메트릭 | 보조 메트릭 |
|------|-------------|-------------|
| ASR | WER(단어 오류율) | CER(문자 오류율) · RTFx(실시간 계수) · 첫 토큰 지연 시간 |
| TTS | MOS(평균 의견 점수) / UTMOS(Unbiased TTS MOS) | SECS(유사성, 표현력, 자연스러움, 안정성) · ASR 왕복 WER · CER(문자 오류율) · TTFA(첫 음성 생성 시간) |
| 음성 복제 | SECS(ECAPA 코사인) | MOS · CER(문자 오류율) |
| 화자 검증 | EER(동일 오류율) | minDCF(최소 검출 비용) · 운영 지점에서의 FAR(거짓 수락률) / FRR(거짓 거부율) |
| 화자 분리 | DER(대화자 오류율) | JER(대화자 오류율) · 화자 혼동 |
| 오디오 분류 | top-1 정확도 · mAP(평균 정밀도) | 매크로 F1 · 클래스별 재현율 |
| 음악 생성 | FAD(Fréchet Audio Distance) | CLAP(Contrastive Language-Audio Pretraining) · 청취 패널 MOS |
| 오디오 언어 모델 | MMAU-Pro(Multimodal Audio Understanding) | LongAudioBench · AudioCaps FENSE |
| 스트리밍 S2S | 지연 시간 P50/P95 | WER(단어 오류율) · MOS(평균 의견 점수) |

## 개념

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

## ASR 평가 지표

**WER (Word Error Rate, 단어 오류율).** `(S + D + I) / N`. 점수 매기기 전에 소문자로 변환, 구두점 제거, 숫자 정규화. `jiwer` 또는 OpenAI의 `whisper_normalizer` 사용. &lt; 5% = 인간 수준 읽기 음성.

**CER (Character Error Rate, 문자 오류율).** 동일한 공식, 문자 수준. 단어 분할이 모호한 성조 언어(만다린, 광둥어)에 사용.

**RTFx (역 실시간 계수).** 벽시계 1초당 처리된 오디오 초. 높을수록 좋음. Parakeet-TDT는 3380×. Whisper-large-v3는 ~30×.

**첫 토큰 지연 시간.** 오디오 입력부터 첫 전사 토큰까지의 벽시계 시간. 스트리밍에 중요. Deepgram Nova-3: ~150 ms.

## TTS 평가 지표

**MOS (Mean Opinion Score, 평균 의견 점수).** 1-5 인간 평가. 표준이지만 느림. 샘플당 20명 이상, 모델당 100개 이상 샘플 수집.

**UTMOS (2022-2026).** 학습된 MOS 예측기. 표준 벤치마크에서 인간 MOS와 ~0.9 상관관계. F5-TTS: UTMOS 3.95; 실제 값: 4.08.

**SECS (Speaker Encoder Cosine Similarity, 화자 인코더 코사인 유사도).** 음성 복제에 사용. 참조와 복제된 출력 간 ECAPA 임베딩 코사인. &gt; 0.75 = 인식 가능한 복제.

**ASR 왕복 WER.** TTS 출력에 Whisper 실행 후 입력 텍스트 대비 WER 계산. 명료도 저하 감지. 2026 SOTA: &lt; 2% CER.

**TTFA (Time-To-First-Audio, 첫 오디오 생성 시간).** 벽시계 지연 시간. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

## 음성 복제 특화

**SECS + MOS + CER** 삼중 평가. SECS는 높지만 MOS가 낮으면 음색은 맞지만 부자연스러운 복제; 반대는 자연스럽지만 화자가 틀린 경우.

## 화자 검증

**EER (Equal Error Rate, 등오류율).** 거짓 수락률과 거짓 거부율이 같은 임계값. VoxCeleb1-O에서 ECAPA: 0.87%.

**minDCF (최소 검출 비용).** 선택한 운영 지점(종종 FAR=0.01)에서의 가중치 비용. EER보다 실제 적용성 높음.

## 화자 분리

**DER (Diarization Error Rate, 화자 분리 오류율).** `(FA + Miss + Confusion) / total_speaker_time`. 놓친 음성 + 오경보 음성 + 화자 혼동, 각각 비율. AMI 회의: DER ~10-20% 현실적. pyannote 3.1 + Precision-2 상용: 잘 녹음된 오디오에서 &lt;10% DER.

**JER (Jaccard Error Rate, 자카드 오류율).** DER 대안, 짧은 세그먼트 편향에 강건.

## 오디오 분류

다중 레이블: **mAP (mean Average Precision, 평균 평균 정밀도)** 모든 클래스 대상. AudioSet: BEATs-iter3의 0.548 mAP.

상호 배타적 다중 클래스: **top-1, top-5 정확도**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

불균형: **매크로 F1** + **클래스별 재현율**. 클래스별 보고 — 집계 정확도는 실패한 클래스 숨김.

## 음악 생성

**FAD (Fréchet Audio Distance, 프레셰 오디오 거리).** 실제 vs 생성 오디오의 VGGish-임베딩 분포 간 거리. MusicCaps에서 MusicGen-small: 4.5. MusicLM: 4.0. 낮을수록 좋음.

**CLAP 점수.** CLAP 임베딩 사용 텍스트-오디오 정렬 점수. &gt; 0.3 = 합리적 정렬.

**청취 패널 MOS.** 소비자 등급 음악의 최종 평가 기준. Suno v5 ELO 1293 (TTS Arena의 쌍별 인간 선호도).

## 오디오-언어 벤치마크

**MMAU (Massive Multi-Audio Understanding, 대규모 다중 오디오 이해).** 10k 오디오-QA 쌍.

**MMAU-Pro.** 1800개 고난도 항목, 4개 범주: 음성 / 소리 / 음악 / 다중 오디오. 4-way에서 무작위 확률 25%. Gemini 2.5 Pro 전체 ~60%; 다중 오디오 ~22% (모든 모델).

**LongAudioBench.** 의미론적 질의가 있는 수 분 길이 클립. Audio Flamingo Next가 Gemini 2.5 Pro를 제침.

**AudioCaps / Clotho.** 캡셔닝 벤치마크. SPICE, CIDEr, FENSE 지표.

## 스트리밍 음성-음성 변환

**지연 시간 P50 / P95 / P99.** 사용자 음성 종료부터 첫 가청 응답까지의 벽시계 시간. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**출력 WER / MOS.**

**Barge-in 반응성.** 사용자 인터럽트부터 어시스턴트 음소거까지의 시간. 목표 &lt; 150 ms.

## 2026 리더보드

| 리더보드 | 트랙 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 영어 + 다국어 + 장문 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 영어 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, 쌍별 투표 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 추론 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 화자 인식 | `voxsrc.github.io` |
| MMAU 음악 서브셋 | 음악 LALM | (MMAU 내) |
| HEAR 벤치마크 | 자기 지도 오디오 | `hearbenchmark.com` |

## 빌드하기

## 단계 1: 정규화된 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

## 단계 2: TTS 왕복 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

## 단계 3: 음성 복제를 위한 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

## 단계 4: 음악 생성을 위한 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

## 단계 5: 화자 검증을 위한 EER (레슨 6과 동일 코드)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 사용 방법

모든 배포에 모델 업데이트 시마다 실행되는 고정된 평가 하네스를 함께 사용하세요. 세 가지 핵심 규칙:

1. **점수 매기기 전 정규화.** 소문자 변환, 구두점 제거, 숫자 확장. 정규화 규칙을 보고하세요.
2. **평균이 아닌 분포 보고.** 지연 시간에 대한 P50/P95/P99. 분류에 대한 클래스별 재현율. MMAU에 대한 카테고리별 결과.
3. **하나의 표준 공개 벤치마크 실행.** 프로덕션 데이터가 다르더라도 Open ASR / TTS Arena / MMAU에 대한 보고를 통해 검토자들이 동일한 기준으로 비교할 수 있게 하세요.

## 함정(Pitfalls)

- **UTMOS 외삽(extrapolation).** VCTK 스타일의 깨끗한 음성으로 훈련됨; 노이즈가 있는 / 복제된 / 감정적인 오디오에 대한 점수 평가가 부정확함.
- **MOS 패널 편향(bias).** 20명의 Amazon Mechanical Turk 작업자 ≠ 20명의 목표 사용자. 중요도가 높다면 도메인별 패널을 구성하라.
- **FAD는 참조 집합에 의존함.** 모델 간 비교 시 동일한 참조 분포를 사용하라.
- **집계된 WER.** 전체 5% WER은 억양 있는 음성에서 30% WER을 숨길 수 있음. 인구통계학적 분할(demographic slice)별로 보고하라.
- **공개 벤치마크 포화 상태.** 대부분의 최첨단 모델은 표준 벤치마크에서 한계점(ceiling)에 근접함. 실제 트래픽을 반영하는 내부 보유(in-house held-out) 데이터셋을 구축하라.

## Ship It

`outputs/skill-audio-evaluator.md`로 저장. 모든 오디오 모델 출시를 위한 메트릭, 벤치마크, 보고 형식을 선정한다.

## 1. 메트릭 선정
- **음질 평가**  
  - PESQ(Perceptual Evaluation of Speech Quality)  
  - STOI(Short-Time Objective Intelligibility)  
  - SI-SDR(Scale-Invariant Signal-to-Distortion Ratio)  

- **음성 인식 정확도**  
  - WER(Word Error Rate)  
  - CER(Character Error Rate)  

- **생성 품질**  
  - FAD(Fréchet Audio Distance)  
  - KL 발산성(Kullback-Leibler Divergence)  

- **다운스트림 작업 성능**  
  - 감정 인식 정확도(Emotion Recognition Accuracy)  
  - 화자 식별 정확도(Speaker Identification Accuracy)  

## 2. 벤치마크 데이터셋
- **음질/음성 인식**  
  - LibriSpeech  
  - TIMIT  
  - VOiCES  

- **생성 품질**  
  - Freesound (배경음/환경음)  
  - AudioSet (다양한 오디오 이벤트)  

- **다운스트림 작업**  
  - IEMOCAP (감정 인식)  
  - VoxCeleb (화자 식별)  

## 3. 보고 형식
| 항목 | 내용 |
|------|------|
| **모델 정보** | 아키텍처, 파라미터 수, 학습 데이터 출처 |
| **평가 결과** | 각 메트릭별 수치 (예: PESQ 3.2, WER 12.5%) |
| **벤치마크 비교** | SOTA 모델 대비 성능 차이 (표/그래프) |
| **실패 사례** | WER > 20%인 샘플, 생성 실패 사례 (오디오 클립 링크) |
| **추론 속도** | 초당 처리 샘플 수(real-time factor) |
| **배포 환경** | 지원 하드웨어(CPU/GPU), 최소 시스템 요구사항 |

## 4. 추가 검증
- **인간 평가**  
  - MOS(Mean Opinion Score) 1~5점 척도 (20명 이상 평가자)  
  - A/B 테스트 (기존 모델 대비 선호도)  

- **강건성 테스트**  
  - 배경 잡음 추가 시 성능 변화 (SNR 0dB, 5dB, 10dB)  
  - 화자/악센트 다양성 테스트  

> 📌 **릴리스 조건**: 모든 메트릭이 베이스라인 대비 5% 이상 향상 또는 인간 평가 MOS ≥ 4.0

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 장난감 입력에 대해 WER(단어 오류율) / CER(문자 오류율) / EER(동등 오류율) / SECS(음성-텍스트 일관성 점수) / FAD-ish(프리치 오디오 거리 유사도) / MMAU-ish(멀티모달 오디오 이해 유사도)를 계산합니다.
2. **중간.** TTS(텍스트-음성 변환) 왕복 WER 평가 도구를 구축합니다. Kokoro 또는 F5-TTS 출력을 Whisper로 변환한 후 50개 프롬프트에 대한 WER을 계산합니다. WER > 10%인 프롬프트를 표시합니다.
3. **어려움.** 레슨 10의 LALM(대형 오디오 언어 모델) 선택을 MMAU-Pro 음성 + 다중 오디오 서브셋(각각 50개 항목)에 대해 평가합니다. 카테고리별 정확도를 보고하고 공개된 수치와 비교합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| WER | ASR 점수 | 정규화 후 단어 수준에서 `(S+D+I)/N`. |
| CER | 문자 수준 WER | 성조 언어 또는 문자 수준 시스템에 사용. |
| MOS | 인간 평가 | 1-5점 평가; 20명 이상의 청취자 × 100개 샘플. |
| UTMOS | ML 기반 MOS 예측기 | 학습된 모델; 인간 MOS와 ~0.9 상관관계. |
| SECS | 음성 복제 유사도 | 참조 음성과 복제 음성 간 ECAPA 코사인 유사도. |
| EER | 화자 검증 점수 | FAR = FRR인 임계값. |
| DER | 화자 분리 점수 | (FA + Miss + Confusion) / 총 발화 수. |
| FAD | 음악 생성 품질 | VGGish 임베딩에 대한 Fréchet 거리. |
| RTFx | 처리량 | 벽시계 1초당 오디오 처리 초. |

> **용어 설명**:  
> - S: 치환(Substitutions), D: 삭제(Deletions), I: 삽입(Insertions), N: 참조 텍스트 단어 수  
> - FA: 거짓 경보(False Alarms), Miss: 누락(Misses), Confusion: 혼동(Confusions)

## 추가 자료

- [jiwer](https://github.com/jitsi/jiwer) — 정규화 유틸리티가 포함된 WER/CER(단어/문자 오류율) 라이브러리.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 학습된 MOS(평균 의견 점수) 예측기.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 음악 생성 표준.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026년 실시간 순위.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 인간 투표 기반 TTS(텍스트-음성 변환) 리더보드.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM(대형 오디오 언어 모델) 추론 리더보드.
- [HEAR benchmark](https://hearbenchmark.com/) — 오디오 SSL(자기 지도 학습) 벤치마크.