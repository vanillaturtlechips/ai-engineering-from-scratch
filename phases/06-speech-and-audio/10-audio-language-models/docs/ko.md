# 오디오-언어 모델 — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026년 오디오-언어 모델은 음성 + 환경 음향 + 음악을 기반으로 추론합니다. Qwen2.5-Omni-7B는 MMAU-Pro에서 GPT-4o Audio와 성능을 일치시킵니다. Audio Flamingo Next는 LongAudioBench에서 Gemini 2.5 Pro를 능가합니다. 오픈 소스와 클로즈드 소스 간 격차는 사실상 해소되었습니다 — 다만 다중 오디오 작업에서는 모든 모델이 무작위 수준에 가깝습니다.

**유형:** 학습  
**언어:** Python  
**선수 지식:** 6단계 · 04 (ASR), 12단계 · 03 (비전-언어 모델), 7단계 · 10 (오디오 트랜스포머)  
**소요 시간:** ~45분

## 문제 정의

5초 분량의 오디오가 있습니다: 개 짖는 소리, 누군가 "멈춰!"라고 외친 후 침묵. 유용한 질문은 여러 축을 아우릅니다:

- **전사(Transcription).** "무슨 말이 있었나요?" — 자동 음성 인식(ASR) 영역.
- **의미적 추론(Semantic reasoning).** "사람이 위험에 처했나요?" — 개 짖는 소리 + 외침 + 침묵에 대한 공동 이해가 필요.
- **음악적 추론(Music reasoning).** "어떤 악기가 멜로디를 연주하나요?"
- **장문 오디오 검색(Long-audio retrieval).** "이 90분 강의 중 어디서 강사가 기울기 하강(gradient descent)을 설명했나요?"

이 모든 질문에 하나의 프롬프트로 답변하는 단일 모델은 **오디오-언어 모델**(LALM / ALM)입니다. 순수 ASR과 구별되는데, LALM은 전사본뿐 아니라 자유로운 자연어 답변을 생성합니다.

## 개념

![오디오-언어 모델: 오디오 인코더 + 프로젝터 + LLM 디코더](../assets/alm-architecture.svg)

### 3-컴포넌트 템플릿

모든 2026 LALM은 동일한 골격을 가집니다:

1. **오디오 인코더.** Whisper 인코더 · BEATs · CLAP · WavLM · 또는 모델별 커스텀 인코더.
2. **프로젝터.** 선형(linear) 또는 MLP로 오디오-인코더 특성을 LLM의 토큰 임베딩 공간으로 변환.
3. **LLM.** Llama / Qwen / Gemma 기반 디코더. 텍스트 + 오디오 토큰을 인터리빙하여 입력받고 텍스트 생성.

훈련:

- **1단계.** 인코더 + LLM 고정; 프로젝터만 ASR / 캡셔닝 데이터로 훈련.
- **2단계.** 전체 / LoRA 파인튜닝을 통해 지시 따르기 오디오 작업(QA, 추론, 음악 이해) 적용.
- **3단계 (선택).** 음성 입력/출력 추가 시 음성 디코더 추가. Qwen2.5-Omni 및 AF3-Chat이 이 방식을 사용.

### 2026 모델 맵

| 모델 | 백본 | 오디오 인코더 | 출력 모달리티 | 접근 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 커스텀 + Whisper | 텍스트 + 음성 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 커스텀 | 텍스트 + 음성 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 텍스트 | NVIDIA 비상업적 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 텍스트 | NVIDIA 비상업적 |
| SALMONN | Vicuna | Whisper + BEATs | 텍스트 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 텍스트 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 텍스트 | Apache-2.0 |
| Gemini 2.5 Flash/Pro (폐쇄형) | Gemini | 프로프라이어터리 | 텍스트 + 음성 | API |
| GPT-4o Audio (폐쇄형) | GPT-4o | 프로프라이어터리 | 텍스트 + 음성 | API |

### 벤치마크 현실 점검 (2026)

**MMAU-Pro.** 음성 / 소리 / 음악 / 혼합 1800개 QA 쌍. 다중 오디오 서브셋 포함.

| 모델 | 전체 | 음성 | 소리 | 음악 | 다중 오디오 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**다중 오디오 열은 모든 모델에 치명적**입니다. 4지선다형 무작위 정답률 = 25%; 대부분 모델은 이 수준 근처 점수. LALM은 여전히 두 클립 비교에 어려움을 겪습니다.

### 2026년 LALM의 유용한 분야

- **콜센터 녹음 준수 감사.** "상담원이 필수 공개 사항을 언급했는가?"
- **접근성.** 청각 장애인에게 소리 이벤트 설명(단순 전사 아님).
- **콘텐츠 조정.** 폭력적 언어 + 위협적 어조 + 배경 맥락 감지.
- **팟캐스트 / 회의 챕터링.** 화자 전환뿐 아니라 의미적 요약.
- **음악 카탈로그 분석.** "B섹션에서 키 변경이 있는 모든 트랙 찾기."

### 아직 유용하지 않은 분야

- 세밀한 음악 이론(코드 수준 이하).
- 긴 대화에서의 화자 귀속 추론(10분 이상 시 성능 저하).
- 다중 오디오 비교(22-26%는 무작위보다 약간 높음).
- 실시간 스트리밍 추론(대부분 오프라인 배치 추론).

## 구축 방법

### 1단계: Qwen2.5-Omni 쿼리하기

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "어떤 소리가 들리고, 무슨 일이 일어나고 있나요?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 2단계: 프로젝터 패턴

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

이게 전부입니다. 프로젝터는 일반적으로 1-3개의 선형 레이어로 구성됩니다. ASR 쌍(오디오 → 전사)을 통해 훈련하는 것이 1단계 사전 작업입니다.

### 3단계: MMAU / LongAudioBench 벤치마킹

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"정확도: {correct / len(mmau['test']):.3f}")
```

카테고리별(음성/소리/음악/다중 오디오) 결과를 별도로 보고하세요. 집계된 수치는 모델의 실패 지점을 숨길 수 있습니다.

## 사용 방법

| 작업 | 2026년 추천 모델 |
|------|------------------|
| 자유 형식 오디오 QA (오픈) | Qwen2.5-Omni-7B |
| 긴 오디오 처리 최고 성능 (오픈) | Audio Flamingo Next |
| 최고 성능 (클로즈드) | Gemini 2.5 Pro |
| 음성 입력/출력 에이전트 | Qwen2.5-Omni 또는 GPT-4o Audio |
| 음악 추론 | Audio Flamingo 3 또는 2 (음악 특화 AF-CLAP) |
| 콜센터 감사 | 정책 문서 RAG를 활용한 Gemini 2.5 Pro API |

## 함정(Pitfalls)

- **다중 오디오(over-trust on multi-audio)에 대한 과도한 신뢰.** 작업에 "어떤 클립에 X가 있는가"가 필요한 경우, 무작위 확률 수준의 성능이 실제로 발생할 수 있습니다.
- **장문 오디오(long-audio degradation) 성능 저하.** 10분을 넘어가면 대부분의 모델에서 화자 식별(speaker attribution)이 실패합니다. 먼저 화자 분할(diarization)을 수행(레슨 6)한 후 요약하세요.
- **무음 구간(hallucinations on silence)에서의 환각 현상.** Whisper 인코더를 사용하는 LALM(Large Audio Language Model)에서 Whisper 스타일의 문제가 그대로 발생합니다. VAD(Voice Activity Detection)로 필터링하세요.
- **벤치마크 체리피킹(benchmark cherry-picking).** 벤더 블로그 게시물은 최상의 사례 범주만 강조합니다. 직접 MMAU-Pro 다중 오디오 서브셋을 실행해 보세요.

## Ship It

`outputs/skill-alm-picker.md`로 저장. 주어진 오디오 이해 작업에 대해 LALM(대형 오디오 언어 모델) + 벤치마크 부분 집합 + 출력 방식(텍스트 vs 음성)을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하여 장난감 프로젝터 패턴 + 가짜 LALM 라우팅(음성-임베딩, 텍스트-토큰) → 출력 토큰을 확인해 보세요.
2. **중간.** Qwen2.5-Omni-7B를 100개의 MMAU-Pro 음성 항목에 대해 평가하세요. 논문에서 보고된 수치와 비교해 보세요.
3. **어려움.** 최소 오디오-캡션 기준 모델을 구축하세요: BEATs 인코더 + 2층 프로젝터 + 고정된 Llama-3.2-1B. AudioCaps에서 프로젝터만 파인튜닝하세요. Clotho-AQA에서 SALMONN과 비교해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| LALM | 오디오 ChatGPT | 오디오 인코더 + 프로젝터 + LLM 디코더. |
| 프로젝터(Projector) | 어댑터(Adapter) | 오디오 특징을 LLM 임베딩 공간으로 매핑하는 소형 MLP. |
| MMAU | 벤치마크 | 음성, 소리, 음악 분야의 10,000개 오디오-QA 쌍. |
| MMAU-Pro | 더 어려운 MMAU | 1,800개의 다중 오디오/추론 중심 질문. |
| LongAudioBench | 장형 평가 | 의미론적 쿼리가 포함된 수 분 길이의 클립. |
| Voice-in / voice-out | 음성 네이티브 | 텍스트 경유 없이 음성을 입력받고 음성을 출력하는 모델. |

## 추가 자료

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 참조 아키텍처.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 음성-음성 변환.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 오픈형 장기 오디오 리더.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 듀얼 인코더 선구자.
- [MMAU-Pro 리더보드](https://mmaubenchmark.github.io/) — 2026년 실시간 순위.