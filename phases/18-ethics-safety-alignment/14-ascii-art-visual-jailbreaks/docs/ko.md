# ASCII 아트와 시각적 탈옥

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: 정렬된 LLM에 대한 ASCII 아트 기반 탈옥 공격" (ACL 2024, arXiv:2402.11753). 유해 요청에서 안전 관련 토큰을 마스킹하고, 동일한 문자의 ASCII 아트 렌더링으로 대체한 후 위장된 프롬프트를 전송합니다. GPT-3.5, GPT-4, Gemini, Claude, Llama-2는 모두 ASCII 아트 토큰을 견고하게 인식하지 못합니다. 이 공격은 PPL(퍼플렉서티 필터), 의역 방어, 재토큰화를 우회합니다. 관련: ViTC 벤치마크는 비의미적 시각적 프롬프트 인식을 측정하며, StructuralSleight는 비일반 텍스트 인코딩 구조(트리, 그래프, 중첩 JSON)를 인코딩 공격 패밀리로 일반화합니다.

**유형:** Build  
**언어:** Python (stdlib, ArtPrompt 토큰 마스킹 하니스)  
**사전 요구 사항:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)  
**소요 시간:** ~60분

## 학습 목표

- **ArtPrompt 공격** 설명: 단어 식별 단계, ASCII-아트 대체, 최종 은폐된 프롬프트.
- **표준 방어 기법(PPL, Paraphrase, Retokenization)**이 ArtPrompt에 실패하는 이유 설명.
- **ViTC** 정의 및 측정 항목 설명.
- **StructuralSleight** 설명: 임의의 비일반 텍스트 인코딩 구조로의 일반화.

## 문제

의역 및 역할극(Lesson 12)과 긴 컨텍스트(Lesson 13)를 통한 공격은 텍스트 수준의 패턴에서 작동합니다. ArtPrompt는 인식 수준에서 작동합니다: 모델은 금지된 토큰을 파싱하지 않습니다. 문자로 렌더링된 이미지를 파싱합니다. 안전 필터는 무해한 구두점을 인식합니다. 모델은 단어를 인식합니다.

## 개념

### ArtPrompt, 두 단계

단계 1. 단어 식별. 유해한 요청이 주어지면 공격자는 LLM을 사용하여 안전 관련 단어(예: "how to make a bomb"에서 "bomb")를 식별합니다. 

단계 2. 위장 프롬프트 생성. 식별된 각 단어를 ASCII 아트 표현(7x5 또는 7x7 문자 블록으로 구성된 글자 모양)으로 대체합니다. 모델은 충분히 강력한 모델이 단어로 인식할 수 있는 구두점과 공백의 그리드를 수신하지만, 안전 필터는 그리드만 인식합니다.

결과: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 모두 실패. 벤치마크 부분집합에서 공격 성공률 75% 이상.

### 표준 방어 기법이 실패하는 이유

- **PPL(퍼플렉서티 필터).** ASCII 아트는 높은 퍼플렉서티를 가지지만, 모든 새로운 입력도 마찬가지입니다. ArtPrompt를 차단하는 임계값 선택은 합법적인 구조화된 입력도 차단합니다.
- **패러프레이징.** 프롬프트를 패러프레이징하면 ASCII 아트가 파괴됩니다. 실제로 패러프레이징 LLM은 종종 아트를 보존하거나 재구성합니다.
- **재토큰화.** 토큰 분할 방식을 다르게 해도 모델의 "시각"이 글자 모양을 인식하는 것은 변하지 않습니다.

근본적인 문제는 안전 필터가 토큰 또는 의미 수준인데, ArtPrompt는 시각적 인식 수준에서 작동한다는 것입니다.

### ViTC 벤치마크

비의미적 시각적 프롬프트 인식. ASCII 아트, 윙딩스 및 기타 비텍스트-의미적 시각적 콘텐츠를 읽는 모델의 능력을 측정합니다. ArtPrompt의 효과는 ViTC 정확도와 상관관계가 있습니다: 모델이 시각적 텍스트를 더 잘 읽을수록 ArtPrompt가 더 잘 작동합니다. 이는 능력-안전 트레이드오프입니다.

### StructuralSleight

ArtPrompt를 일반화: 흔하지 않은 텍스트 인코딩 구조(UTES). 트리, 그래프, 중첩된 JSON, JSON 내 CSV, diff 스타일 코드 블록. 훈련 안전 데이터에서는 드물지만 모델이 파싱할 수 있는 구조라면 유해한 콘텐츠를 숨길 수 있습니다.

방어적 함의: 안전은 모델이 파싱할 수 있는 구조화된 표현 전반에 걸쳐 일반화되어야 합니다. 이 집합은 크고 계속 성장합니다.

### 이미지 모달리티 유사 사례

시각적 LLM(GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1)은 공격 표면을 확장합니다. 실제 이미지를 사용한 ArtPrompt 스타일 공격은 ASCII 아트 유사 사례보다 더 강력합니다. 이미지 인코더가 더 풍부한 신호를 생성하기 때문입니다.

### 18단계에서의 위치

레슨 12-14는 세 가지 직교 공격 벡터(반복적 개선(PAIR), 컨텍스트 길이(MSJ), 인코딩(ArtPrompt/StructuralSleight))를 설명합니다. 레슨 15는 모델 중심 공격에서 시스템 경계 공격(간접 프롬프트 인젝션)으로 전환합니다. 레슨 16은 방어 도구 응답을 설명합니다.

## 사용 방법

`code/main.py`는 장난감 ArtPrompt를 구축합니다. 유해한 쿼리에서 특정 단어를 ASCII 아트 글리프(glyph)로 숨기고, 숨겨진 문자열이 키워드 필터를 통과하는지 검증하며, (선택적으로) 간단한 인식기를 사용해 숨겨진 문자열을 다시 디코딩할 수 있습니다.

## Ship It

이 레슨은 `outputs/skill-encoding-audit.md`를 생성합니다. 탈옥 방어 보고서를 기반으로, (ASCII 아트, base64, 리트-스피크(leet-speak), UTF-8 동형 문자(homoglyph), UTES) 인코딩 공격 패밀리와 각 공격을 탐지하는 방어 계층을 열거합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 가려진 문자열이 간단한 키워드 필터를 통과하는지 확인하세요. 문자 수준 변경 요구 사항을 보고하세요.

2. 동일한 대상 단어에 대해 두 번째 인코딩(베이스64)을 구현하세요. ArtPrompt 대비 필터 우회율과 복구 난이도를 비교하세요.

3. Jiang et al. 2024 4.3절(5개 모델 결과)을 읽으세요. 동일한 벤치마크에서 Claude의 ArtPrompt 저항성이 Gemini보다 높은 이유를 제안하세요.

4. 프롬프트 내 ASCII-아트 형태 영역을 탐지하는 사전 생성 방어 기법을 설계하세요. 합법적 코드, 표, 수학 표기법에 대한 오탐률을 측정하세요.

5. StructuralSleight는 10가지 인코딩 구조를 나열합니다. 10가지 모두를 처리하는 일반화된 방어 기법을 스케치하고, 방어된 프롬프트당 계산 비용을 추정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| ArtPrompt | "the ASCII-art attack" | 안전 단어를 ASCII 아트 렌더링으로 마스킹하는 2단계 탈옥 기법 |
| Cloaking | "hide the word" | 금지된 토큰을 필터는 읽지 못하지만 모델은 읽을 수 있는 시각적 표현으로 대체 |
| UTES | "uncommon structure" | 비일반 텍스트 인코딩 구조(트리, 그래프, 중첩 JSON 등) — 콘텐츠 밀반입에 사용 |
| ViTC | "visual-text capability" | 비의미적 시각 인코딩을 읽는 모델의 능력을 평가하는 벤치마크 |
| Perplexity filter | "PPL defense" | 높은 퍼플렉서티(perplexity)를 가진 프롬프트 거부; 합법적 구조화 입력도 높은 점수를 받아 실패 |
| Retokenization | "tokenizer shift defense" | 다른 토크나이저로 프롬프트를 사전 처리; 인식이 시각적이므로 실패 |
| Homoglyph | "lookalike characters" | 라틴 문자와 동일하게 보이는 유니코드 문자; 부분 문자열 검사 우회 |

## 추가 자료

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII-아트 탈옥 논문
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 일반화
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 보완적 반복 공격
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 보완적 길이 공격