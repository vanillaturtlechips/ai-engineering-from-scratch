# 서브워드 토크나이저 — BPE, WordPiece, Unigram, SentencePiece

> 단어 토크나이저는 처음 보는 단어에서 막히고, 문자 토크나이저는 시퀀스 길이를 폭발적으로 증가시킵니다. 서브워드 토크나이저는 이 둘의 중간을 선택합니다. 모든 현대 LLM은 서브워드 토크나이저를 기반으로 합니다.

**유형:** 학습
**언어:** Python
**선수 지식:** Phase 5 · 01 (텍스트 처리), Phase 5 · 04 (GloVe / FastText / 서브워드)
**소요 시간:** ~60분

## 문제

당신의 어휘 사전에는 50,000개의 단어가 있습니다. 사용자가 "untokenizable"을 입력하면 토크나이저는 `[UNK]`를 반환합니다. 이제 모델은 해당 단어에 대한 어떤 신호도 받지 못합니다. 더 나쁜 것은: 코퍼스에서 90번째 백분위수 문서는 40개의 희귀 단어를 포함하는데, 이는 문서당 40비트의 정보 손실이 발생함을 의미합니다.

서브워드 토크나이제는 이 문제를 해결합니다. 일반적인 단어는 단일 토큰으로 유지됩니다. 희귀 단어는 의미 있는 조각으로 분해됩니다: `untokenizable` → `un`, `token`, `izable`. 모든 문자열은 궁극적으로 바이트 시퀀스이므로 훈련 데이터는 모든 것을 포괄합니다.

2026년의 모든 최첨단 LLM(Large Language Model)은 세 가지 알고리즘(BPE, Unigram, WordPiece) 중 하나로 구현되고, 세 가지 라이브러리(tiktoken, SentencePiece, HF Tokenizers) 중 하나로 래핑되어 출시됩니다. 당신은 이 중 하나를 선택하지 않고는 언어 모델을 출시할 수 없습니다.

## 개념

![BPE vs Unigram vs WordPiece, 문자별 비교](../assets/subword-tokenization.svg)

**BPE(Byte-Pair Encoding).** 문자 수준 어휘로 시작. 모든 인접 쌍을 카운트. 가장 빈도가 높은 쌍을 새 토큰으로 병합. 목표 어휘 크기에 도달할 때까지 반복. 주요 알고리즘: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**바이트 수준 BPE.** 동일한 알고리즘이지만 유니코드 문자 대신 원시 바이트(256개 기본 토큰) 사용. `[UNK]` 토큰이 전혀 발생하지 않음 — 모든 바이트 시퀀스 인코딩 가능. GPT-2는 50,257개 토큰(256개 바이트 + 50,000개 병합 + 1개 특수) 사용.

**유니그램(Unigram).** 거대한 어휘로 시작. 각 토큰에 유니그램 확률 할당. 코퍼스 로그 우도(log-likelihood)를 가장 적게 증가시키는 토큰을 반복적으로 제거. 추론 시 확률적: 토큰화 샘플링 가능(서브워드 정규화를 통한 데이터 증강에 유용). T5, mBART, ALBERT, XLNet, Gemma에서 사용.

**워드피스(WordPiece).** 원시 빈도 대신 훈련 코퍼스의 우도를 최대화하는 쌍을 병합. BERT, DistilBERT, ELECTRA에서 사용.

**SentencePiece vs tiktoken.** SentencePiece는 원시 유니코드 텍스트에서 직접 어휘(BPE 또는 유니그램)를 *훈련*하는 라이브러리이며, 공백을 `▁`로 인코딩. tiktoken은 미리 구축된 어휘에 대한 OpenAI의 고속 *인코더*로, 훈련 기능은 없음.

경험적 규칙:

- **새 어휘 훈련:** SentencePiece(다국어, 사전 토큰화 불필요) 또는 HF Tokenizers.
- **GPT 어휘에 대한 고속 추론:** tiktoken(cl100k_base, o200k_base).
- **둘 다:** HF Tokenizers — 단일 라이브러리로 훈련 + 서빙.

## 구축 방법

### 1단계: 처음부터 BPE 학습

`code/main.py` 참조. 다음 루프:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

알고리즘이 인코딩하는 세 가지 사실. `</w>`는 단어 끝을 표시하여 "low"(접미사)와 "lower"(접두사)가 구분되도록 합니다. 빈도 가중치는 고빈도 쌍이 초기에 승리하도록 합니다. 병합 목록은 순서가 있으며, 추론은 학습 순서대로 병합을 적용합니다.

### 2단계: 학습된 병합으로 인코딩

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

순진한 O(n·|merges|) 구현. 프로덕션 구현(tiktoken, HF Tokenizers)은 우선순위 큐를 사용한 병합 순위 조회를 통해 거의 선형 시간에 실행됩니다.

### 3단계: 실제 SentencePiece 사용

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 또는 "unigram"
    character_coverage=0.9995, # CJK의 경우 더 낮게 설정 (예: 영어 0.9995, 일본어 0.995)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

주의: 사전 토크나이징이 필요 없으며, 공백은 `▁`로 인코딩됩니다. `character_coverage`는 희귀 문자를 얼마나 공격적으로 보존할지 또는 ``로 매핑할지 제어합니다.

### 4단계: OpenAI 호환 어휘를 위한 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

인코딩 전용. 빠른 성능(Rust 백엔드). GPT-4/5 토크나이징과 정확히 일치하여 바이트 수 계산, 비용 추정, 컨텍스트 윈도우 예산 책정에 사용됩니다.

## 2026년에도 여전히 발생하는 함정

- **토큰화기 드리프트.** 어휘 집합 A로 학습, 어휘 집합 B로 배포. 토큰 ID가 달라 모델 출력이 무의미해짐. CI에서 `tokenizer.json` 해시를 확인.
- **공백 모호성.** BPE "hello" vs " hello"는 다른 토큰을 생성. 항상 `add_special_tokens`와 `add_prefix_space`를 명시적으로 지정.
- **다국어 학습 부족.** 영어 중심 코퍼스는 비라틴 문자를 5-10배 더 많은 토큰으로 분할하는 어휘 집합을 생성. GPT-3.5에서 일본어/아랍어 프롬프트는 동일한 프롬프트에 대해 5-10배 더 많은 비용이 발생. o200k_base가 이 문제를 부분적으로 해결.
- **이모지 분할.** 단일 이모지가 5개의 토큰을 차지할 수 있음. 컨텍스트 예산 책정 시 체크포인트의 이모지 처리 방식 확인.

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| 단일 언어 모델을 처음부터 훈련 | HF 토크나이저(BPE) |
| 다국어 모델 훈련 | SentencePiece(Unigram, `character_coverage=0.9995`) |
| OpenAI 호환 API 제공 | tiktoken(GPT-4+용 `o200k_base`) |
| 도메인 특화 어휘(코드, 수학, 단백질) | 도메인 코퍼스로 커스텀 BPE 훈련, 기본 어휘와 병합 |
| 엣지 추론, 소형 모델 | Unigram(더 작은 어휘 집합이 더 잘 작동) |

어휘 크기는 확장성 결정 사항이며 상수가 아닙니다. 대략적인 휴리스틱: 매개변수 1B 미만은 32k, 1-10B는 50-100k, 다국어/프론티어 모델은 200k+.

```mermaid
graph TD
    A[모델 훈련] --> B{단일 언어?}
    B -->|예| C[HF Tokenizers (BPE)]
    B -->|아니오| D[SentencePiece (Unigram)]
    A --> E[API 제공]
    E --> F[tiktoken (`o200k_base`)]
    A --> G[도메인 특화]
    G --> H[커스텀 BPE + 기본 어휘 병합]
    A --> I[엣지 추론]
    I --> J[Unigram (소형 어휘)]
```

## Ship It

`outputs/skill-tokenizer-picker.md`로 저장:

```markdown
---
name: tokenizer-picker
description: 주어진 코퍼스와 배포 대상에 대해 토크나이저 알고리즘, 어휘 크기, 라이브러리를 선택합니다.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

코퍼스(크기, 언어, 도메인)와 배포 대상(처음부터 학습 / 파인튜닝 / API 호환 추론)이 주어졌을 때 다음을 출력합니다:

1. 알고리즘. BPE, Unigram, 또는 WordPiece. 한 문장 이유.
2. 라이브러리. SentencePiece, HF Tokenizers, 또는 tiktoken. 이유.
3. 어휘 크기. 가장 가까운 1k 단위로 반올림. 모델 크기와 언어 커버리지와 연관된 이유.
4. 커버리지 설정. `character_coverage`, `byte_fallback`, 특수 토큰 목록.
5. 검증 계획. 평균 토큰-단어 비율(held-out 세트), OOV(Out-Of-Vocabulary) 비율, 압축 비율, 왕복 디코딩 일치 여부.

희귀한 스크립트 내용이 포함된 코퍼스에 대해 `character_coverage` <0.995인 토크나이저 학습을 거부합니다. CI에서 동결된 `tokenizer.json` 해시 체크 없이 어휘를 배포하는 것을 거부합니다. 16k 미만 어휘의 단일 언어 토크나이저는 사양 미달 가능성이 높다고 표시합니다.
```

## 연습 문제

1. **쉬움.** `code/main.py`의 작은 코퍼스에 대해 500-merge BPE를 학습시키세요. 3개의 홀드아웃 단어를 인코딩하세요. 정확히 1개의 토큰을 생성한 경우와 >1개의 토큰을 생성한 경우는 각각 몇 개인가요?
2. **중간.** 100개의 영어 위키피디아 문장에 대해 `cl100k_base`, `o200k_base`, 그리고 vocab=32k로 학습시킨 SentencePiece BPE 간의 토큰 수를 비교하세요. 각각의 압축 비율을 보고하세요.
3. **어려움.** 동일한 코퍼스에 대해 BPE, Unigram, WordPiece를 학습시키세요. 각각의 방법을 작은 감성 분류기에 적용했을 때 다운스트림 정확도를 측정하세요. 선택에 따라 F1 점수가 1점 이상 변동하나요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| BPE | 바이트-페어 인코딩(Byte-Pair Encoding) | 목표 어휘 크기(vocab size)에 도달할 때까지 가장 빈번한 문자 쌍을 탐욕적으로(greedy) 병합하는 방식. |
| 바이트-레벨 BPE(Byte-level BPE) | 알 수 없는 토큰(unknown token)이 절대 없음 | 원시 256바이트(raw 256 bytes)에 BPE 적용; GPT-2 / Llama에서 사용. |
| 유니그램(Unigram) | 확률적 토크나이저(Probabilistic tokenizer) | 로그-우도(log-likelihood)를 사용해 대규모 후보 집합에서 가지치기(pruning) 수행; T5, Gemma에서 사용. |
| 센텐스피스(SentencePiece) | 공백(whitespace) 처리 방식 | 원시 텍스트에서 BPE/유니그램을 학습시키는 라이브러리; 공백은 `▁`로 인코딩. |
| 틱토큰(tiktoken) | 빠른 토크나이저 | 사전 구축된 어휘(vocab)에 대해 OpenAI의 Rust 기반 BPE 인코더; 학습 불필요. |
| 병합 목록(Merge list) | 마법 같은 숫자 | `(a, b) → ab` 형태의 순서화된 병합 목록; 추론 시 순서대로 적용. |
| 문자 커버리지(Character coverage) | 얼마나 희귀해야 너무 희귀한가? | 토크나이저가 반드시 커버해야 하는 훈련 코퍼스의 문자 비율; 일반적으로 ~0.9995. |

## 추가 자료

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 논문.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 논문.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 라이브러리.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 간결한 참고 자료.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 쿡북 + 인코딩 목록.