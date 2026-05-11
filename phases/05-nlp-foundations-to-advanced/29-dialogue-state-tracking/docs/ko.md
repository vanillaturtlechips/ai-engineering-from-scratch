# 대화 상태 추적(Dialogue State Tracking)

> "북쪽에 있는 저렴한 식당을 원해... 아, 중간 가격으로 바꿔줘... 그리고 이탈리아식 추가해줘." 세 번의 대화, 세 번의 상태 업데이트. DST는 슬롯-값 사전을 동기화하여 예약이 원활히 진행되도록 합니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** 5단계 · 17(챗봇), 5단계 · 20(구조화된 출력)
**소요 시간:** ~75분

## 문제 정의

작업 지향 대화 시스템에서 사용자의 목표는 슬롯-값 쌍 집합으로 인코딩됩니다: `{cuisine: italian, area: north, price: moderate}`. 모든 사용자 발화는 슬롯을 추가, 변경 또는 제거할 수 있습니다. 시스템은 전체 대화를 읽고 현재 상태를 정확히 출력해야 합니다.

단일 슬롯을 잘못 처리하면 시스템이 잘못된 레스토랑을 예약하거나, 잘못된 항공편을 예약하거나, 잘못된 카드로 결제할 수 있습니다. DST(Dialogue State Tracking)는 사용자가 말한 내용과 백엔드에서 실행되는 내용 사이의 연결고리입니다.

LLM 시대에도 2026년에 여전히 중요한 이유:

- 규정 준수 민감 분야(은행, 의료, 항공 예약)는 자유 형식 생성이 아닌 결정적 슬롯 값을 요구합니다.
- 도구 사용 에이전트는 API 호출 전 슬롯 해결이 여전히 필요합니다.
- 다중 턴 수정은 생각보다 어렵습니다: "아니, 목요일로 바꿔줘."

현대 파이프라인: 고전적 DST 개념 + LLM 추출기 + 구조화된 출력 가드레일.

## 개념

![DST: 대화 기록 → 슬롯-값 상태](../assets/dst.svg)

**작업 구조.** 스키마는 도메인(레스토랑, 호텔, 택시)과 해당 슬롯(요리 유형, 지역, 가격, 인원 수)을 정의합니다. 각 슬롯은 비어 있거나, 닫힌 집합에서 값을 가질 수 있습니다(가격: {저렴, 중간, 고가}), 또는 자유 형식 값을 가질 수 있습니다(이름: "The Copper Kettle").

**두 가지 DST 공식화.**

- **분류.** 각 (슬롯, 후보_값) 쌍에 대해 예/아니오를 예측합니다. 닫힌 어휘 슬롯에 적용됩니다. 2020년 이전의 표준 방식입니다.
- **생성.** 대화를 기반으로 슬롯 값을 자유 텍스트로 생성합니다. 열린 어휘 슬롯에 적용됩니다. 현대적인 기본 방식입니다.

**평가 지표.** 통합 목표 정확도(JGA) — *모든* 슬롯이 정확한 턴의 비율입니다. 전부 또는 무(all-or-nothing) 방식입니다. MultiWOZ 2.4 리더보드는 2026년 기준 약 83%를 기록했습니다.

**아키텍처.**

1. **규칙 기반(슬롯 정규식 + 키워드).** 좁은 도메인에 강력한 기준선. 디버깅 가능.
2. **TripPy / BERT-DST.** BERT 인코딩을 사용한 복사 기반 생성. LLM 이전 표준.
3. **LDST(LLaMA + LoRA).** 도메인-슬롯 프롬프팅이 적용된 지시 튜닝 LLM. MultiWOZ 2.4에서 ChatGPT 수준의 품질 달성.
4. **온톨로지 프리(2024–26).** 스키마를 생략; 슬롯 이름과 값을 직접 생성. 열린 도메인 처리 가능.
5. **프롬프트 + 구조화된 출력(2024–26).** Pydantic 스키마 + 제약된 디코딩이 적용된 LLM. 5줄의 코드로 프로덕션 준비 완료.

### 고전적인 실패 사례

- **턴 간 공참조.** "첫 번째 옵션으로 진행합시다." 어떤 옵션을 가리키는지 해결해야 합니다.
- **덮어쓰기 vs 추가.** 사용자가 "이탈리아식 추가"라고 말할 때, 요리 유형을 대체해야 할까요, 추가해야 할까요?
- **암묵적 확인.** "좋아요" — 제안된 예약을 수락한 것인가요?
- **수정.** "사실 7시로 변경해 주세요." 다른 슬롯을 지우지 않고 시간만 업데이트해야 합니다.
- **이전 시스템 발화 참조.** "네, 그 거요." "그 거"는 무엇을 가리키나요?

## 구축 방법

### 1단계: 규칙 기반 슬롯 추출기

`code/main.py` 참조. 정규 표현식 + 동의어 사전은 좁은 도메인에서 표준 발화의 70%를 커버:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

표준 어휘 범위 외에서는 취약. 결정적 슬롯 확인에는 작동.

### 2단계: 상태 업데이트 루프

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

세 가지 불변 사항:

- 사용자가 건드리지 않은 슬롯은 절대 재설정하지 않음.
- 명시적 부정("cuisine은 신경 쓰지 마세요")은 반드시 클리어해야 함.
- 사용자 수정("사실은...")은 덮어쓰기, 추가하지 않음.

### 3단계: 구조화된 출력을 통한 LLM 기반 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""턴마다 레스토랑 예약 슬롯 값을 추적합니다.
지금까지의 대화:
{render(history)}

최신 사용자 발화를 기반으로 상태를 업데이트합니다. JSON 상태만 출력하세요."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic은 유효한 상태 객체를 보장. 정규 표현식 불필요, 스키마 불일치 없음, 환각 슬롯 없음.

### 4단계: JGA 평가

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

보정: 시스템이 모든 슬롯을 정확히 맞추는 턴 비율은? MultiWOZ 2.4 기준 2026년 상위 시스템: 80-83%. 좁은 어휘에서 인-도메인 시스템이 이를 초과하거나 LLM 기준선이 더 높아야 함.

### 5단계: 수정 처리

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

수정 감지 시, 마지막 업데이트된 슬롯을 덮어쓰기. LLM 없이는 정확히 구현하기 어려움. 현대적 패턴: 증분 업데이트 대신 항상 LLM이 전체 상태를 재생성하도록 함 — 이는 수정을 자연스럽게 처리.

## 함정

- **전체 기록 재생성 비용.** 매 턴마다 LLM이 상태를 재생성하도록 하면 총 O(n²) 토큰 비용이 발생합니다. 기록을 제한하거나 이전 턴을 요약하세요.
- **스키마 드리프트.** 사후에 새로운 슬롯을 추가하면 이전 훈련 데이터가 깨집니다. 스키마 버전을 관리하세요.
- **대소문자 구분.** "Italian" vs "italian" vs "ITALIAN" — 모든 곳에서 정규화하세요.
- **암묵적 상속.** 사용자가 이전에 "4인분"을 지정했다면, 다른 시간에 대한 새 요청이 인원 수를 초기화해서는 안 됩니다. 항상 전체 기록을 전달하세요.
- **자유 형식 vs 폐쇄 집합.** 이름, 시간, 주소는 자유 형식 슬롯이 필요하며, 요리 유형과 지역은 폐쇄 집합입니다. 스키마에 두 유형을 혼합하세요.

## 사용 방법

2026 스택:

| 상황 | 접근 방식 |
|-----------|----------|
| 좁은 도메인 (하나 또는 두 개의 의도) | 규칙 기반 + 정규 표현식(Regex) |
| 넓은 도메인, 레이블된 데이터 사용 가능 | LDST (LLaMA + MultiWOZ 스타일 데이터의 LoRA) |
| 넓은 도메인, 레이블 없음, 프로덕션 준비 완료 | LLM + Instructor + Pydantic 스키마 |
| 음성 / 보이스 | ASR + 정규화기 + LLM-DST |
| 다중 도메인 예약 흐름 | 스키마 기반 LLM + 도메인별 Pydantic 모델 |
| 규정 준수 민감 | 규칙 기반 주, 확인 흐름이 있는 LLM 대체 |

## Ship It

`outputs/skill-dst-designer.md`로 저장:

```markdown
---
name: dst-designer
description: 대화 상태 추적기(dialogue state tracker) 설계 — 스키마, 추출기, 업데이트 정책, 평가.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

사용 사례(도메인, 언어, 어휘 개방성, 규정 준수 요구사항)가 주어지면 다음을 출력:

1. 스키마. 도메인 목록, 도메인별 슬롯, 슬롯별 개방/폐쇄 어휘.
2. 추출기. 규칙 기반 / seq2seq / LLM-with-Pydantic. 근거.
3. 업데이트 정책. 전체 상태 재생성 / 증분; 수정 처리; 부정 처리.
4. 평가. 보류된 대화 세트에서의 Joint Goal Accuracy, 슬롯 수준 정밀도/재현율, 가장 어려운 슬롯에 대한 혼동.
5. 확인 흐름. 사용자에게 명시적으로 확인을 요청해야 하는 시기(파국적 작업, 낮은 신뢰도 추출).

규정 준수 민감 슬롯에 대해 규칙 기반 2차 검증 없이 LLM-only DST를 거부. 사용자 수정 시 슬롯 롤백이 불가능한 모든 DST를 거부. 버전 태그가 없는 스키마를 플래그 처리.
```

## 연습 문제

1. **쉬움.** `code/main.py`에 3개의 슬롯(요리 유형(cuisine), 지역(area), 가격(price))에 대한 규칙 기반 상태 추적기(rule-based state tracker)를 구현하세요. 10개의 수작업 대화(hand-crafted dialogues)로 테스트하고 JGA(공동 목표 달성률)를 측정하세요.
2. **중간.** 동일한 데이터셋에 Instructor + Pydantic + 소형 LLM을 적용하세요. JGA를 비교하고 가장 어려운 턴(turn)을 분석하세요.
3. **어려움.** 두 시스템을 모두 구현하고 라우팅 전략을 적용하세요: 규칙 기반(rule-based)을 주력으로 사용하고, 규칙 기반이 신뢰도(confidence) <2 슬롯을 출력할 때 LLM을 대체(fallback)로 사용합니다. 결합된 JGA와 턴당 추론 비용(inference cost per turn)을 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| DST | 대화 상태 추적(Dialogue State Tracking) | 대화 턴 간에 슬롯-값 사전(slot-value dict)을 유지합니다. |
| 슬롯(Slot) | 사용자 의도의 단위 | 백엔드에서 필요한 명명된 파라미터(예: 음식 종류(cuisine), 날짜(date)). |
| 도메인(Domain) | 작업 영역 | 레스토랑(restaurant), 호텔(hotel), 택시(taxi) — 슬롯들의 집합. |
| JGA | 결합 목표 정확도(Joint Goal Accuracy) | 모든 슬롯이 정확한 턴의 비율. 전부 아니면 무(all-or-nothing). |
| MultiWOZ | 벤치마크 | 다중 도메인 WOZ 데이터셋; 표준 DST 평가. |
| 온톨로지-프리 DST(Ontology-free DST) | 스키마 없음 | 고정된 목록 없이 슬롯 이름과 값을 직접 생성합니다. |
| 수정(Correction) | "사실은..." | 이전에 채워진 슬롯을 덮어쓰는 턴.

## 추가 자료

- [Budzianowski et al. (2018). MultiWOZ — 대규모 다중 도메인 마법사-오즈](https://arxiv.org/abs/1810.00278) — 표준 벤치마크.
- [Feng et al. (2023). LLM 기반 대화 상태 추적(LDST) 방향](https://arxiv.org/abs/2310.14970) — DST를 위한 LLaMA + LoRA 지시 튜닝.
- [Heck et al. (2020). TripPy — 값 독립적 신경망 대화 상태 추적을 위한 삼중 복사 전략](https://arxiv.org/abs/2005.02877) — 복사 기반 DST 작업 핵심.
- [King, Flanigan (2024). LLM을 활용한 비지도 종단간 작업 지향 대화](https://arxiv.org/abs/2404.10753) — EM 기반 비지도 TOD.
- [MultiWOZ 리더보드](https://github.com/budzianowski/multiwoz) — 표준 DST 결과.