# 처음부터 토크나이저 구축하기

> 레슨 01에서는 장난감을 제공했습니다. 이번 레슨에서는 무기를 제공합니다.

**유형:** 구축
**언어:** Python
**선수 과목:** Phase 10, 레슨 01 (토크나이저: BPE, WordPiece, SentencePiece)
**소요 시간:** ~90분

## 학습 목표

- 유니코드, 공백 정규화, 특수 토큰을 처리하는 프로덕션급 BPE 토크나이저 구축
- 바이트 수준 폴백 구현으로 이모지, CJK 문자, 코드 등 모든 입력을 알 수 없는 토큰 없이 인코딩할 수 있도록 지원
- BPE 병합 적용 전 단어 경계에서 텍스트를 분할하는 사전 토크나이징 정규식 패턴 추가
- 코퍼스에 맞춤형 토크나이저 훈련 및 다국어 텍스트에서 tiktoken 대비 압축률 평가 수행

## 문제

Lesson 01의 BPE 토크나이저는 영어 텍스트에서 작동합니다. 이제 일본어나 이모지, 또는 탭과 공백이 혼합된 Python 코드를 입력해 보세요.

깨집니다.

BPE가 잘못되어서가 아니라 구현이 불완전하기 때문입니다. 프로덕션용 토크나이저는 모든 인코딩의 원시 바이트를 처리하고, 분할 전 유니코드를 정규화하며, 절대 병합되지 않는 특수 토큰을 관리하고, 사전 토크나이징과 서브워드 분할을 연결하며, 15조 개의 토큰을 처리하는 학습 파이프라인의 병목 현상이 발생하지 않을 만큼 빠르게 이 모든 작업을 수행합니다.

GPT-2의 토크나이저는 50,257개의 토큰을 가지고 있습니다. Llama 3은 128,256개, GPT-4는 대략 100,000개입니다. 이는 장난감이 아닌 실제 규모입니다. 이 어휘집들의 병합 테이블은 수백 기가바이트의 텍스트로 학습되었으며, 주변 메커니즘(정규화, 사전 토크나이징, 특수 토큰 주입, 채팅 템플릿 포맷팅 등)이 "hello world"를 처리하는 토크나이저와 전체 인터넷을 처리하는 토크나이저를 구분합니다.

이제 그 메커니즘을 구축할 것입니다.

## 개념

### 전체 파이프라인

프로덕션 토크나이저는 하나의 알고리즘이 아닙니다. 각기 다른 문제를 해결하는 5단계 파이프라인으로 구성됩니다.

```mermaid
graph LR
    A[원시 텍스트] --> B[정규화]
    B --> C[사전 토크나이징]
    C --> D[BPE 병합]
    D --> E[특수 토큰]
    E --> F[토큰 ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

각 단계는 특정 역할을 수행합니다:

| 단계 | 수행 작업 | 중요성 |
|-------|-------------|----------------|
| 정규화 | NFKC 유니코드, 소문자화 선택 사항, 악센트 제거 선택 사항 | "fi" 리가처(U+FB01)가 "fi"(두 문자)로 변환됩니다. 이 단계가 없으면 같은 단어가 다른 토큰을 얻습니다. |
| 사전 토크나이징 | BPE 전에 텍스트를 청크로 분할 | BPE가 단어 경계를 넘어 병합하는 것을 방지합니다. "the cat"은 절대 "e c" 토큰을 생성해서는 안 됩니다. |
| BPE 병합 | 바이트 시퀀스에 학습된 병합 규칙 적용 | 핵심 압축. 원시 바이트를 서브워드 토큰으로 변환합니다. |
| 특수 토큰 | [BOS], [EOS], [PAD], 채팅 템플릿 마커 삽입 | 이 토큰들은 고정 ID를 가집니다. BPE 병합에 절대 참여하지 않습니다. 모델은 구조를 위해 이 토큰들이 필요합니다. |
| ID 매핑 | 토큰 문자열을 정수 ID로 변환 | 모델은 문자열이 아닌 정수를 인식합니다. |

### 바이트 수준 BPE

레슨 01의 토크나이저는 UTF-8 바이트에서 작동했습니다. 이는 올바른 선택이었습니다. 하지만 중요한 것을 생략했습니다: 해당 바이트가 유효한 UTF-8이 아닐 때 어떻게 될까요?

바이트 수준 BPE는 모든 가능한 바이트 값(0-255)을 유효한 토큰으로 처리하여 이 문제를 해결합니다. 기본 어휘집은 정확히 256개 항목입니다. 텍스트, 바이너리, 손상된 파일 등 모든 파일을 알 수 없는 토큰을 생성하지 않고 토크나이징할 수 있습니다.

GPT-2는 트릭을 추가했습니다: 각 바이트를 인쇄 가능한 유니코드 문자로 매핑하여 어휘집을 사람이 읽을 수 있게 유지합니다. 바이트 0x20(공백)은 매핑에서 문자 "G"가 됩니다. 이는 순전히 미관적인 목적입니다. 알고리즘은 신경 쓰지 않습니다.

진정한 강점: 바이트 수준 BPE는 지구상의 모든 언어를 처리합니다. 한자는 각각 3개의 UTF-8 바이트입니다. 일본어는 3-4바이트일 수 있습니다. 아랍어, 데바나가리, 이모지 — 모두 바이트 시퀀스일 뿐입니다. BPE 알고리즘은 영어 ASCII 바이트에서 패턴을 찾는 것과 정확히 같은 방식으로 이 바이트 시퀀스에서 패턴을 찾습니다.

### 사전 토크나이징

BPE가 텍스트에 닿기 전에 청크로 분할해야 합니다. 이는 병합 알고리즘이 단어 경계를 넘는 토큰을 생성하는 것을 방지합니다.

GPT-2는 정규식 패턴을 사용하여 텍스트를 분할합니다:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

이 패턴은 축약형("don't"는 "don" + "'t"로 분할), 선택적 선행 공백이 있는 단어, 숫자, 구두점, 공백을 기준으로 분할합니다. 선행 공백은 단어에 붙어 있습니다 — 따라서 "the cat"은 [" the", " cat"]이 되며 ["the", " ", "cat"]이 아닙니다.

Llama는 SentencePiece를 사용하며, 정규식을 완전히 건너뜁니다. 원시 바이트 스트림을 하나의 긴 시퀀스로 처리하고 BPE 알고리즘이 경계를 파악하도록 합니다. 이는 더 간단하지만 BPE가 단어 간 토큰을 생성할 수 있는 더 많은 자유를 줍니다.

선택은 중요합니다. GPT-2의 정규식은 토크나이저가 한 단어의 끝에 있는 "the"와 다음 단어의 시작에 있는 "the"를 병합하는 것을 학습하지 못하게 합니다. SentencePiece는 이를 허용하여 때로는 더 효율적인 압축을 생성하지만 덜 해석 가능한 토큰을 생성합니다.

### 특수 토큰

모든 프로덕션 토크나이저는 구조적 마커를 위한 토큰 ID를 예약합니다:

| 토큰 | 목적 | 사용 모델 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 시퀀스 시작 | Llama 3, GPT |
| `[EOS]` / `</s>` | 시퀀스 종료 | 모든 모델 |
| `[PAD]` | 배치 정렬을 위한 패딩 | BERT, T5 |
| `[UNK]` | 알 수 없는 토큰(바이트 수준 BPE는 이를 제거) | BERT, WordPiece |
| `<\|im_start\|>` | 채팅 메시지 경계 시작 | ChatGPT, Qwen |
| `<\|im_end\|>` | 채팅 메시지 경계 종료 | ChatGPT, Qwen |
| `<\|user\|>` | 사용자 턴 마커 | Llama 3 |
| `<\|assistant\|>` | 어시스턴트 턴 마커 | Llama 3 |

특수 토큰은 BPE에 의해 절대 분할되지 않습니다. 병합 알고리즘이 실행되기 전에 정확히 매칭되어 고정 ID로 대체되고, 주변 텍스트는 정상적으로 토크나이징됩니다.

### 채팅 템플릿

대부분의 사람들이 혼란스러워하고 대부분의 구현이 실패하는 부분입니다.

채팅 모델에 메시지를 보낼 때 API는 메시지 목록을 허용합니다:

```
[
  {"role": "system", "content": "당신은 도움이 됩니다."},
  {"role": "user", "content": "안녕하세요"},
  {"role": "assistant", "content": "안녕하세요!"}
]
```

모델은 JSON을 보지 않습니다. 평평한 토큰 시퀀스를 봅니다. 채팅 템플릿은 특수 토큰을 사용하여 메시지를 이 평평한 시퀀스로 변환합니다. 모든 모델은 이를 다르게 처리합니다:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 도움이 됩니다.<|eot_id|><|start_header_id|>user<|end_header_id|>

안녕하세요<|eot_id|><|start_header_id|>assistant<|end_header_id|>

안녕하세요!<|eot_id|>

ChatGPT:
system
당신은 도움이 됩니다. 
user
안녕하세요
assistant
안녕하세요!
```

템플릿을 잘못 사용하면 모델이 쓰레기 출력을 생성합니다. 모델은 하나의 정확한 형식으로 훈련되었습니다. 누락된 새 줄, 바뀐 토큰, 여분의 공백 등 어떤 편차도 입력을 훈련 분포 밖으로 밀어냅니다.

### 속도

Python은 프로덕션 토크나이징에 너무 느립니다.

tiktoken(OpenAI)은 Python 바인딩이 있는 Rust로 작성되었습니다. HuggingFace 토크나이저도 Rust입니다. SentencePiece는 C++입니다. 이들은 순수 Python보다 10-100배 속도 향상을 달성합니다.

참고: 15조 토큰을 Llama 3 사전 훈련에 사용하는 경우, 초당 100만 토큰(빠른 Python) 처리 시 174일이 소요됩니다. 초당 1억 토큰(Rust) 처리 시 1.7일이 걸립니다.

알고리즘을 이해하기 위해 Python으로 구축합니다. 프로덕션에서는 컴파일된 구현을 사용하고 Python 래퍼만 건드립니다.

## 구축 방법

### 1단계: 바이트 수준 인코딩

기초입니다. 모든 문자열을 바이트 시퀀스로 변환하고, 각 바이트를 표시용 인쇄 가능 문자에 매핑한 후 역변환합니다.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

다국어 텍스트로 테스트하여 바이트 수를 확인합니다:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello"는 5바이트입니다. "你好"는 6바이트(각 문자당 3바이트)입니다. 불 이모지는 4바이트입니다. 바이트 수준 토크나이저는 언어를 구분하지 않습니다. 바이트는 바이트일 뿐입니다.

### 2단계: 정규 표현식 기반 사전 토크나이저

GPT-2 정규 표현식 패턴을 사용하여 텍스트를 청크로 분할합니다. 각 청크는 BPE에 의해 독립적으로 토큰화됩니다.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 모듈은 유니코드 속성 이스케이프(`\p{L}`은 문자, `\p{N}`은 숫자)를 지원합니다. 표준 라이브러리 `re` 모듈은 지원하지 않으므로 ASCII 문자 클래스로 대체합니다. 프로덕션 다국어 토크나이저의 경우 `regex`를 설치하세요.

테스트해 봅니다:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

선행 공백은 단어에 붙어 있습니다. 축약형은 아포스트로피에서 분할됩니다. 구두점은 별도의 청크가 됩니다. BPE는 이 경계를 넘어 토큰을 병합하지 않습니다.

### 3단계: 바이트 시퀀스에 대한 BPE

레슨 01의 핵심 알고리즘이지만, 이제 사전 토큰화된 청크를 독립적으로 처리합니다.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 4단계: 특수 토큰 처리

특수 토큰은 정확한 매칭과 고정 ID가 필요합니다. BPE를 완전히 우회합니다.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 5단계: 전체 토크나이저 클래스

모든 것을 연결합니다: 정규화, 특수 토큰 분할, 사전 토큰화, BPE 병합, ID 매핑.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 6단계: 다국어 테스트

실제 테스트입니다. 영어, 중국어, 이모지, 코드를 처리합니다.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

한자는 각각 3바이트를 생성합니다. 이모지는 4바이트를 생성합니다. 이 중 어느 것도 토크나이저를 충돌시키지 않습니다. 알 수 없는 토큰도 생성되지 않습니다. 이것이 바이트 수준 BPE의 힘입니다.

## 사용 방법

### 실제 토크나이저 비교

Llama 3, GPT-4, Mistral의 실제 토크나이저를 로드합니다. 각 토크나이저가 동일한 다국어 단락을 어떻게 처리하는지 확인해 보세요.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

동일한 텍스트에 대해 다른 토큰 수가 나타나는 것을 확인할 수 있습니다. 128K 어휘를 가진 Llama 3은 일반적인 패턴을 더 적극적으로 병합합니다. 100K 어휘를 가진 GPT-4는 중간 수준에 위치합니다. 32K 어휘를 가진 Mistral은 더 많은 토큰을 생성하지만 임베딩 레이어가 더 작습니다.

트레이드오프는 항상 동일합니다: 더 큰 어휘는 더 짧은 시퀀스를 의미하지만 더 많은 파라미터를 필요로 합니다.

## Ship It

이 레슨은 프로덕션 토크나이저를 구축하고 디버깅하기 위한 프롬프트를 생성합니다. `outputs/prompt-tokenizer-builder.md`를 참조하세요.

## 연습 문제

1. **쉬움:** `get_token_bytes(id)` 메서드를 추가하세요. 이 메서드는 모든 토큰 ID에 대한 원시 바이트를 보여줍니다. 이를 사용하여 가장 일반적인 병합된 토큰이 실제로 무엇을 나타내는지 검사하세요.
2. **중간:** 공백과 숫자에서 분할하지만 선행 공백을 유지하는 Llama 스타일 사전 토크나이저를 구현하세요. 동일한 코퍼스에서 GPT-2 정규식 접근법과 어휘를 비교하세요.
3. **어려움:** `{"role": ..., "content": ...}` 메시지 목록을 입력으로 받아 Llama 3 채팅 형식에 맞는 올바른 토큰 시퀀스를 생성하는 채팅 템플릿 메서드를 추가하세요. HuggingFace 구현과 비교하여 테스트하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 바이트 수준 BPE | "바이트 단위로 작동하는 토크나이저" | 256개의 바이트 값을 기본 어휘로 사용하는 BPE -- 알 수 없는 토큰 없이 모든 입력 처리 가능 |
| 사전 토크나이징 | "BPE 이전의 분할" | 단어 경계를 넘어 병합하지 않도록 방지하는 정규식 또는 규칙 기반 분할 |
| NFKC 정규화 | "유니코드 정리" | 표준 분해 후 호환성 결합 -- "fi" 리가처는 "fi"로, 전각 "A"는 "A"로 변환 |
| 채팅 템플릿 | "메시지가 토큰이 되는 방식" | 역할/내용 메시지 목록을 평탄한 토큰 시퀀스로 변환하는 정확한 형식 -- 모델별 고유하며 학습 형식과 일치해야 함 |
| 특수 토큰 | "제어 토큰" | BPE를 우회하는 예약된 토큰 ID -- [BOS], [EOS], [PAD], 채팅 마커 -- 병합 전 정확히 일치해야 함 |
| 생산성 | "단어당 토큰 수" | 출력 토큰과 입력 단어의 비율 -- GPT-4에서 영어는 1.3, 한국어는 2-3, 높을수록 컨텍스트 낭비 |
| tiktoken | "OpenAI 토크나이저" | Python 바인딩이 있는 Rust BPE 구현 -- 순수 Python보다 10-100배 빠름 |
| 병합 테이블 | "어휘" | 학습 중 학습된 바이트-페어 병합 순서 목록 -- 이것이 토크나이저의 학습된 지식 그 자체 |

## 추가 자료

- [OpenAI tiktoken 소스](https://github.com/openai/tiktoken) -- GPT-3.5/4에서 사용하는 Rust BPE 구현
- [HuggingFace 토크나이저](https://github.com/huggingface/tokenizers) -- BPE, WordPiece, Unigram을 지원하는 Rust 토크나이저 라이브러리
- [Llama 3 논문 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 어휘 및 토크나이저 학습 세부 정보
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 언어-불문 토큰화
- [GPT-2 토크나이저 소스](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 원본 바이트-유니코드 매핑