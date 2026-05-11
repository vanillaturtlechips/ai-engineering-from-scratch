# 텍스트를 위한 CNN과 RNN

> 컨볼루션은 n-gram을 학습합니다. 순환은 기억합니다. 둘 다 어텐션에 의해 대체되었지만, 제한된 하드웨어에서는 여전히 중요합니다.

**유형:** 구축
**언어:** Python
**선수 지식:** 3단계 · 11 (PyTorch 입문), 5단계 · 03 (단어 임베딩), 4단계 · 02 (기초부터 시작하는 컨볼루션)
**소요 시간:** ~75분

## 문제 정의

TF-IDF와 Word2Vec은 단어 순서를 무시하는 평탄한 벡터를 생성했습니다. 이를 기반으로 구축된 분류기는 `dog bites man`과 `man bites dog`를 구분하지 못했습니다. 단어 순서는 때때로 중요한 신호를 담고 있습니다.

트랜스포머가 등장하기 전에 이 간극을 메운 두 가지 아키텍처 계열이 있었습니다.

**텍스트를 위한 합성곱 신경망(TextCNN).** 단어 임베딩 시퀀스에 1D 합성곱을 적용합니다. 너비 3의 필터는 학습 가능한 삼단어(트라이그램) 탐지기 역할을 합니다: 세 단어에 걸쳐 점수를 출력합니다. 다양한 너비(2, 3, 4, 5)를 쌓아 다중 스케일 패턴을 탐지합니다. 최대 풀링을 통해 고정 크기 표현으로 변환합니다. 평탄하고 병렬적이며 빠른 처리가 가능합니다.

**순환 신경망(RNN, LSTM, GRU).** 토큰을 하나씩 처리하며, 정보를 앞으로 전달하는 은닉 상태를 유지합니다. 순차적 처리, 메모리 보유, 유연한 입력 길이 지원이 특징입니다. 2014년부터 2017년까지 시퀀스 모델링을 주도했으며, 이후 어텐션 메커니즘이 등장했습니다.

이 강의에서는 두 아키텍처를 모두 구축한 후, 어텐션 메커니즘을 촉발시킨 한계를 설명합니다.

## 개념

![TextCNN 필터 vs. RNN 은닉 상태 펼치기](./assets/cnn-rnn.svg)

**TextCNN** (Kim, 2014). 토큰은 임베딩(embedding)된다. 너비-`k` 1D 컨볼루션(convolution)이 임베딩의 연속된 `k`-그램 위에 필터를 슬라이드하여 특징 맵(feature map)을 생성한다. 해당 맵에 대한 글로벌 최대 풀링(global max-pooling)은 가장 강한 활성화(activation)를 선택한다. 여러 필터 너비에서 나온 최대 풀링 출력을 연결(concatenate)한다. 분류기 헤드(classifier head)에 입력한다.

작동 원리. 필터는 학습 가능한 n-그램이다. 최대 풀링은 위치 불변(position-invariant)이므로 "not good"은 리뷰의 시작이나 중간에서 동일한 특징을 활성화한다. 100개의 필터를 가진 3가지 필터 너비는 300개의 학습된 n-그램 감지기(detector)를 제공한다. 학습은 병렬로 진행되며 순차적 의존성(sequential dependency)이 없다.

**RNN.** 각 시간 단계 `t`에서 은닉 상태 `h_t = f(W * x_t + U * h_{t-1} + b)`이다. `W`, `U`, `b`는 시간 축을 따라 공유된다. 시간 `T`에서의 은닉 상태는 전체 접두사(prefix)의 요약이다. 분류를 위해 `h_1 ... h_T`를 풀링(최대, 평균, 또는 마지막)한다.

일반 RNN은 소실 그래디언트(vanishing gradient) 문제가 있다. **LSTM**은 망각할 것, 저장할 것, 출력할 것을 결정하는 게이트(gate)를 추가하여 긴 시퀀스를 통과하는 그래디언트를 안정화한다. **GRU**는 LSTM을 2개의 게이트로 단순화하며, 더 적은 파라미터로 유사한 성능을 낸다.

**양방향 RNN(Bidirectional RNN)**은 하나는 정방향, 다른 하나는 역방향으로 RNN을 실행하고 은닉 상태를 연결(concatenate)한다. 모든 토큰의 표현은 좌우 컨텍스트(left/right context)를 모두 참조한다. 태깅 작업(tagging task)에 필수적이다.

## 구축 방법

### 1단계: PyTorch에서 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`는 `[batch, seq_len, embed_dim]`을 `[batch, embed_dim, seq_len]`으로 재구성합니다. `nn.Conv1d`는 중간 축을 채널로 처리하기 때문입니다. 풀링된 출력은 입력 길이와 관계없이 고정 크기입니다.

### 2단계: LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

시퀀스 최대 풀링(마지막 상태 풀링 아님). 분류 작업에서는 긴 시퀀스의 끝 정보가 마지막 상태를 지배하는 경향이 있어 최대 풀링이 일반적으로 더 나은 성능을 보입니다.

### 3단계: 소실 기울기 데모(직관)

게이트가 없는 일반 RNN은 장기 의존성을 학습할 수 없습니다. 장난감 작업을 고려해 보세요: 시퀀스 어딘가에 토큰 `A`가 나타났는지 예측. `A`가 위치 1에 있고 시퀀스 길이가 100이면 손실에서 역전파된 기울기는 99번의 순환 가중치 곱셈을 거쳐야 합니다. 가중치가 1보다 작으면 기울기가 소실되고, 1보다 크면 발산합니다.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 가중치=0.9, 100단계:
#   0.9 ^ 100 ≈ 2.7e-5
# 100단계에서 1단계까지의 기울기는 사실상 0입니다.
```

LSTM은 **셀 상태**로 이 문제를 해결합니다. 셀 상태는 덧셈 상호작용만 일어나며 네트워크를 통과합니다(망각 게이트가 곱셈적으로 스케일링하지만 기울기는 여전히 "고속도로"를 따라 흐릅니다). GRU는 더 적은 매개변수로 유사한 작업을 수행합니다. 둘 다 100+ 단계 시퀀스를 통해 안정적인 학습을 가능하게 합니다.

### 4단계: 여전히 부족했던 이유

LSTM을 사용해도 세 가지 문제가 지속되었습니다.

1. **순차적 병목 현상.** 길이 1000의 시퀀스로 RNN을 학습시키려면 1000번의 직렬 순방향/역방향 단계가 필요합니다. 시간 축으로 병렬화할 수 없습니다.
2. **인코더-디코더 설정의 고정 크기 컨텍스트 벡터.** 디코더는 인코더의 최종 은닉 상태만 보며, 이는 전체 입력에 대해 압축된 것입니다. 긴 입력은 세부 정보가 손실됩니다. 레슨 09에서 직접 다룹니다.
3. **원격 의존성 정확도 한계.** LSTM은 일반 RNN보다 성능이 우수하지만 200+ 단계를 넘어 특정 정보를 전파하는 데 여전히 어려움을 겪습니다.

어텐션은 이 세 가지 문제를 모두 해결했습니다. 트랜스포머는 순환 구조를 완전히 제거했습니다. 레슨 10이 전환점입니다.

## 사용 시기

PyTorch의 `nn.LSTM`, `nn.GRU`, `nn.Conv1d`는 프로덕션 환경에서 사용 가능합니다. 학습 코드는 표준 방식을 따릅니다.

Hugging Face는 사전 훈련된 임베딩을 제공하며, 이를 입력 레이어로 바로 사용할 수 있습니다:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

제약 조건에 맞는 사용 시기 체크리스트:

- **에지/온디바이스 추론.** GloVe 임베딩을 사용한 TextCNN은 트랜스포머보다 10-100배 더 작습니다. 배포 대상이 휴대폰이라면 이 스택을 선택하세요.
- **스트리밍/온라인 분류.** RNN은 한 번에 하나의 토큰을 처리하지만, 트랜스포머는 전체 시퀀스가 필요합니다. 실시간으로 들어오는 텍스트의 경우 LSTM이 여전히 유리합니다.
- **기준선용 소형 모델.** 새로운 작업에 대한 빠른 반복. CPU에서 5분 안에 TextCNN을 학습시킬 수 있습니다.
- **제한된 데이터의 시퀀스 라벨링.** BiLSTM-CRF(레슨 06)는 1,000-10,000개의 라벨된 문장에 대해 여전히 프로덕션 등급의 NER 아키텍처입니다.

그 외 모든 경우에는 트랜스포머를 사용합니다.

## Ship It

`outputs/prompt-text-encoder-picker.md`로 저장:

```markdown
---
name: text-encoder-picker
description: 주어진 제약 조건 세트에 대해 텍스트 인코더 아키텍처를 선택합니다.
phase: 5
lesson: 08
---

주어진 제약 조건(작업, 데이터 양, 지연 시간 예산, 배포 대상, 계산 예산)에 따라 다음을 출력합니다:

1. 인코더 아키텍처: TextCNN, BiLSTM, BiLSTM-CRF, 트랜스포머 파인튜닝, 또는 "사전 학습된 트랜스포머를 고정된 인코더로 사용 + 작은 헤드".
2. 임베딩 입력: 무작위 초기화, GloVe / fastText 고정, 또는 컨텍스트화된 트랜스포머 임베딩.
3. 5줄 요약 훈련 레시피: 옵티마이저, 학습률, 배치 크기, 에포크, 정규화.
4. 하나의 모니터링 신호. RNN/CNN 모델의 경우: 어텐션 메커니즘 부재는 장거리 의존성 누락을 의미; 길이별 정확도 확인. 트랜스포머의 경우: 학습률이 너무 높으면 파인튜닝 붕괴; 훈련 손실 확인.

레이블이 지정된 예시가 ~500개 미만일 때 트랜스포머 파인튜닝을 권장하지 않습니다. 단, TextCNN / BiLSTM 베이스라인이 정체기에 도달했음을 보여주는 경우는 예외입니다. 엣지 배포는 아키텍처-우선 접근이 필요함을 표시합니다.
```

## 연습 문제

1. **쉬움.** 3-클래스 장난감 데이터셋(데이터를 직접 생성)에 TextCNN을 학습시켜 보세요. 필터 너비(2, 3, 4)가 단일 너비(3)보다 평균 F1에서 더 우수함을 검증하세요.
2. **중간.** LSTM 분류기를 위해 max-pool, mean-pool, last-state 풀링을 구현하세요. 소규모 데이터셋에서 비교하고, 어떤 풀링이 더 우수한지 문서화하며 그 이유를 가설화하세요.
3. **어려움.** BiLSTM-CRF 개체명 인식(NER) 태거를 구축하세요(레슨 06과 이번 레슨 결합). CoNLL-2003 데이터셋으로 학습시키고, 레슨 06의 CRF 단독 베이스라인과 BERT 파인튜닝과 비교하세요. 학습 시간, 메모리 사용량, F1 점수를 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| TextCNN | 텍스트를 위한 CNN | 단어 임베딩 위에 1D 컨볼루션 스택과 글로벌 최대 풀링 적용. Kim (2014). |
| RNN | 순환 신경망 | 각 시간 단계에서 은닉 상태 업데이트: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | 게이트 순환 신경망 | 입력/망각/출력 게이트 + 셀 상태 추가. 긴 시퀀스를 통해 안정적으로 학습. |
| GRU | 간소화된 LSTM | 3개 대신 2개 게이트. 유사 정확도, 더 적은 파라미터. |
| Bidirectional | 양방향 | 순방향 + 역방향 RNN 연결. 모든 토큰이 컨텍스트 양쪽을 참조. |
| Vanishing gradient | 학습 신호 소멸 | 일반 RNN에서 <1 가중치 반복 곱셈으로 초기 단계 그래디언트가 실질적으로 0이 됨. |

## 추가 학습 자료

- [Kim, Y. (2014). 문장 분류를 위한 합성곱 신경망](https://arxiv.org/abs/1408.5882) — TextCNN 논문. 8페이지. 읽기 쉬움.
- [Hochreiter, S. and Schmidhuber, J. (1997). 장기 단기 메모리](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 논문. 예상외로 명료함.
- [Olah, C. (2015). LSTM 네트워크 이해](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM을 대중적으로 접근 가능하게 만든 다이어그램.