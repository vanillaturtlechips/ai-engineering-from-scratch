# 시퀀스-투-시퀀스 모델

> 번역기를 흉내 내는 두 RNN. 그들이 맞닥뜨린 병목 현상이 어텐션(attention)이 존재하게 된 이유입니다.

**유형:** 구현
**언어:** Python
**선수 지식:** 5단계 · 08 (텍스트 처리를 위한 CNN + RNN), 3단계 · 11 (PyTorch 입문)
**소요 시간:** ~75분

## 문제 정의

분류(classification)는 가변 길이 시퀀스를 단일 레이블로 매핑합니다. 번역(translation)은 가변 길이 시퀀스를 다른 가변 길이 시퀀스로 매핑합니다. 입력과 출력은 서로 다른 어휘 집합에 속하며, 아마도 다른 언어일 수 있으며, 길이 일치에 대한 보장이 없습니다.

seq2seq 아키텍처(Sutskever, Vinyals, Le, 2014)는 의도적으로 단순한 접근법으로 이 문제를 해결했습니다. 두 개의 RNN을 사용합니다. 하나는 소스 문장을 읽고 고정 크기의 컨텍스트 벡터를 생성합니다. 다른 하나는 해당 벡터를 읽고 토큰 단위로 타겟 문장을 생성합니다. 레슨 08에서 작성한 코드와 동일한 구조를 다르게 결합한 것입니다.

이것은 두 가지 이유로 연구할 가치가 있습니다. 첫째, 컨텍스트 벡터 병목 현상은 NLP에서 가장 교육적으로 유용한 실패 사례입니다. 이는 어텐션(attention)과 트랜스포머(transformer)가 잘하는 모든 것을 동기부여합니다. 둘째, 훈련 레시피(teacher forcing, scheduled sampling, 추론 시 빔 서치(beam search))는 LLM을 포함한 모든 현대 생성 시스템에 여전히 적용됩니다.

## 개념

![컨텍스트 벡터 병목 현상을 가진 인코더-디코더](./assets/seq2seq.svg)

**인코더.** 소스 문장을 읽는 RNN. 최종 은닉 상태는 **컨텍스트 벡터** — 전체 입력에 대한 고정 크기 요약. 소스 정보는 잃지 않는다고 가정.

**디코더.** 컨텍스트 벡터로 초기화된 또 다른 RNN. 각 단계에서 이전에 생성된 토큰을 입력으로 받아 타겟 어휘 집합에 대한 분포를 생성. 다음 토큰을 선택하기 위해 샘플링 또는 argmax 사용. 이를 다시 입력으로 공급. `<EOS>` 토큰이 생성되거나 최대 길이에 도달할 때까지 반복.

**훈련:** 각 디코더 단계에서의 교차 엔트로피 손실을 시퀀스 전체에 걸쳐 합산. 표준 역전파(backpropagation through time)를 통해 두 네트워크를 학습.

**티처 포싱(teacher forcing).** 훈련 중에는 디코더의 단계 `t` 입력이 디코더의 이전 예측이 아닌 위치 `t-1`의 *실제 정답* 토큰. 이는 훈련을 안정화함. 티처 포싱이 없으면 초기 오류가 누적되어 모델이 학습하지 못함. 추론 시에는 모델 자체의 예측을 사용해야 하므로 항상 훈련/추론 분포 간 격차가 존재. 이 격차를 **노출 편향(exposure bias)** 이라 함.

**병목 현상.** 인코더가 소스에 대해 학습한 모든 정보는 단일 컨텍스트 벡터로 압축되어야 함. 긴 문장은 세부 정보를 잃음. 희귀 단어는 흐릿해짐. 순서 변경(예: "chat noir" vs. "black cat")은 계산이 아닌 암기로 처리되어야 함.

어텐션(10강)은 디코더가 마지막 은닉 상태가 아닌 *모든* 인코더 은닉 상태를 참조하도록 함으로써 이 문제를 해결. 이것이 핵심 아이디어.

## 구축 단계

### 1단계: 인코더

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`의 형태는 `[batch, seq_len, hidden_dim]`입니다. 입력 위치마다 하나의 은닉 상태가 있습니다. `hidden`의 형태는 `[1, batch, hidden_dim]`입니다. 마지막 단계를 나타냅니다. 레슨 08에서는 "분류를 위해 outputs를 풀링하라"고 했습니다. 여기서는 마지막 은닉 상태를 컨텍스트 벡터로 유지하고 단계별 출력은 무시합니다.

### 2단계: 디코더

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

디코더는 한 번에 한 단계씩 호출됩니다. 입력: 단일 토큰 배치와 현재 은닉 상태. 출력: 다음 토큰에 대한 어휘 로짓과 업데이트된 은닉 상태.

### 3단계: 티처 포싱을 사용한 훈련 루프

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

주목할 만한 두 가지 설정. `ignore_index=0`은 패딩 토큰에 대한 손실을 건너뜁니다. `teacher_forcing_ratio`는 각 단계에서 실제 토큰을 사용할 확률입니다. 모델 예측 대신 사용합니다. 1.0(완전 티처 포싱)에서 시작하여 훈련 중 약 0.5로 점진적으로 줄여 노출-편향 격차를 줄입니다.

### 4단계: 추론 루프(탐욕적)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

탐욕적 디코딩은 모든 단계에서 가장 높은 확률의 토큰을 선택합니다. 이는 잘못된 방향으로 흐를 수 있습니다. 한 번 토큰을 선택하면 되돌릴 수 없습니다. **빔 서치**는 상위 `k`개의 부분 시퀀스를 유지하고 마지막에 가장 높은 점수의 완성된 시퀀스를 선택합니다. 빔 너비 3-5가 일반적입니다.

### 5단계: 병목 현상 시연

토이 복사 작업에서 모델을 훈련시킵니다. 소스 `[a, b, c, d, e]`, 타겟 `[a, b, c, d, e]`. 시퀀스 길이를 늘립니다. 정확도를 관찰합니다.

```
seq_len=5   복사 정확도: 98%
seq_len=10  복사 정확도: 91%
seq_len=20  복사 정확도: 62%
seq_len=40  복사 정확도: 23%
```

단일 GRU 은닉 상태는 40토큰 입력을 손실 없이 기억할 수 없습니다. 정보는 모든 인코더 단계에 있지만 디코더는 마지막 상태만 봅니다. 어텐션은 이를 직접 해결합니다.

## 사용 방법

PyTorch에는 `nn.Transformer`와 `nn.LSTM` 기반 seq2seq 템플릿이 있습니다. Hugging Face의 `transformers` 라이브러리는 수십억 개의 토큰으로 학습된 전체 인코더-디코더 모델(BART, T5, mBART, NLLB)을 제공합니다.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

현대 인코더-디코더는 RNN을 트랜스포머로 대체했습니다. 인코더, 디코더, 토큰별 생성이라는 고수준 구조는 2014년 seq2seq 논문과 동일합니다. 각 블록 내부 메커니즘은 다릅니다.

### 여전히 RNN 기반 seq2seq를 고려해야 할 경우

새 프로젝트에서는 거의 없습니다. 특정 예외 사항:

- 입력 토큰을 한 번에 하나씩 처리하면서 메모리 제한이 있는 스트리밍 번역.
- 트랜스포머의 메모리 비용이 부담되는 온디바이스 텍스트 생성.
- 교육. 인코더-디코더 병목 현상을 이해하는 것이 트랜스포머 원리를 이해하는 가장 빠른 길입니다.

### 노출 편향과 그 해결 방법

- **스케줄드 샘플링.** 훈련 중 티처 포싱 비율을 점진적으로 줄여 모델이 자신의 실수를 복구하는 방법을 학습하도록 합니다.
- **최소 위험 훈련.** 토큰 수준 교차 엔트로피 대신 문장 수준 BLEU 점수로 훈련합니다. 실제 목표에 더 가깝습니다.
- **강화 학습 미세 조정.** 메트릭으로 시퀀스 생성기를 보상합니다. 현대 LLM RLHF에서 사용됩니다.

이 세 가지 방법은 모두 트랜스포머 기반 생성에도 적용됩니다.

## Ship It

`outputs/prompt-seq2seq-design.md`로 저장:

```markdown
---
name: seq2seq-design
description: 주어진 작업에 대한 시퀀스-투-시퀀스 파이프라인을 설계합니다.
phase: 5
lesson: 09
---

작업(번역, 요약, 다른 표현, 질문 재구성)이 주어졌을 때 다음을 출력하세요:

1. 아키텍처. 사전 훈련된 트랜스포머 인코더-디코더(BART, T5, mBART, NLLB)를 기본으로 합니다. 특정 제약 조건이 있을 경우에만 RNN 기반 시퀀스-투-시퀀스를 사용합니다.
2. 시작 체크포인트. 이름(`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`)을 지정합니다. 체크포인트를 작업과 언어 범위에 맞게 선택하세요.
3. 디코딩 전략. 결정적 출력에는 그리디, 품질 향상에는 빔 서치(너비 4-5), 다양성 확보에는 온도 기반 샘플링을 사용합니다. 한 문장으로 정당화하세요.
4. 출시 전 확인할 하나의 실패 모드. 노출 편향은 긴 출력에서 생성 드리프트로 나타납니다. 90% 백분위수 길이의 출력 20개를 샘플링하여 직접 확인하세요.

병렬 예시가 백만 개 미만인 경우 시퀀스-투-시퀀스 모델 처음부터 훈련을 권장하지 않습니다. 사용자 대상 콘텐츠에 그리디 디코딩을 사용하는 파이프라인은 취약하다고 표시하세요(그리디는 반복과 루프를 발생시킵니다).
```

## 연습 문제

1. **쉬움.** 장난감 복사 작업을 구현하세요. 소스와 대상이 동일한 입력-출력 쌍에 대해 GRU seq2seq를 훈련하세요. 길이 5, 10, 20에서의 정확도를 측정하고 병목 현상을 재현하세요.
2. **중간.** 빔 너비 3으로 빔 서치 디코딩을 추가하세요. 작은 병렬 코퍼스에서 탐욕적(greedy) 방식 대비 BLEU 점수를 측정하세요. 빔 서치가 우수한 경우(일반적으로 마지막 토큰)와 차이가 없는 경우를 문서화하세요.
3. **어려움.** 10k 쌍의 패러프레이즈 데이터셋으로 `facebook/bart-base`를 파인튜닝(fine-tuning)하세요. 보류된 입력에 대해 파인튜닝된 모델의 빔-4 출력과 기본 모델의 출력을 비교하세요. BLEU 점수를 보고하고 질적 예시 10개를 선택하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 인코더(Encoder) | 입력 RNN | 소스 입력을 읽습니다. 단계별 은닉 상태와 최종 컨텍스트 벡터를 생성합니다. |
| 디코더(Decoder) | 출력 RNN | 컨텍스트 벡터로 초기화됩니다. 대상 토큰을 하나씩 생성합니다. |
| 컨텍스트 벡터(Context vector) | 요약 | 최종 인코더 은닉 상태. 고정 크기. 어텐션(attention)이 해결하는 병목 지점입니다. |
| 티처 포싱(Teacher forcing) | 실제 토큰 사용 | 훈련 시 실제 이전 토큰을 입력합니다. 학습을 안정화합니다. |
| 노출 편향(Exposure bias) | 훈련/테스트 격차 | 실제 토큰으로 훈련된 모델은 자신의 오류를 복구하는 연습을 하지 못합니다. |
| 빔 서치(Beam search) | 더 나은 디코딩 | 탐욕적으로 선택하는 대신 각 단계에서 상위 k개 부분 시퀀스를 유지합니다.

## 추가 자료

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 최초의 seq2seq 논문. 4페이지 분량.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRU와 인코더-디코더 프레임워크를 소개.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 어텐션(attention) 논문. 이 강의 직후 바로 읽기 권장.
- [PyTorch NLP from Scratch 튜토리얼](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — seq2seq + 어텐션 구현 가능한 코드.