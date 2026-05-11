# Self-Refine 및 CRITIC: 반복적 출력 개선

> Self-Refine (Madaan et al., 2023)은 하나의 LLM을 생성(generate), 피드백(feedback), 개선(refine) 세 가지 역할로 반복적으로 사용하는 방식입니다. 평균 성능 향상: 7개 작업에서 +20 절대 점수. CRITIC (Gou et al., 2023)은 외부 도구를 통한 검증(verification) 라우팅으로 피드백 단계를 강화합니다. 2026년에는 이 패턴이 모든 프레임워크에 "평가자-최적화기"(Anthropic) 또는 가드레일 루프(OpenAI Agents SDK) 형태로 탑재될 예정입니다.

**유형:** 구축(Build)  
**언어:** Python (표준 라이브러리)  
**사전 요구 사항:** 14단계 · 01 (에이전트 루프), 14단계 · 03 (리플렉션)  
**소요 시간:** ~60분

## 학습 목표

- Self-Refine의 세 가지 프롬프트(생성(generate), 피드백(feedback), 개선(refine))를 설명하고, 개선(refine) 프롬프트에 역사(history)가 중요한 이유를 설명하시오.
- CRITIC의 핵심 통찰: 외부 근거(grounding) 없이는 LLM이 자기 검증(self-verification)에 신뢰할 수 없다는 점을 설명하시오.
- 역사(history)와 선택적 외부 검증기(verifier)를 포함한 stdlib Self-Refine 루프를 구현하시오.
- 이 패턴을 Anthropic의 "평가자-최적화기(evaluator-optimizer)" 워크플로우와 OpenAI Agents SDK의 출력 가드레일(output guardrails)에 매핑하시오.

## 문제 정의

에이전트가 거의 정답에 가까운 답변을 생성합니다. 예를 들어 코드 라인에 문법 오류가 있을 수 있고, 요약이 너무 길 수도 있으며, 계획에서 엣지 케이스를 놓칠 수도 있습니다. 필요한 것은 에이전트가 자신의 출력을 스스로 비평한 후 수정하는 것입니다.

Self-Refine은 단일 모델, 학습 데이터 없음, 강화 학습(RL) 없이도 이 방법이 작동함을 보여줍니다. 하지만 함정이 있습니다. LLM은 사실 확인에 약합니다. CRITIC은 해결책을 제시합니다. 검증 단계를 외부 도구(검색, 코드 인터프리터, 계산기, 테스트 러너)로 라우팅하는 것입니다.

이 두 논문은 2026년 반복 개선의 기본 방식을 정의합니다: 생성 → (가능한 경우 외부) 검증 → 개선 → 검증기 통과 시 중단.

## 개념

### Self-Refine (Madaan et al., NeurIPS 2023)

하나의 LLM, 세 가지 역할:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
피드백이 "문제 없음"을 반환하거나 예산이 소진될 때 중지.
```

핵심 세부 사항: `refine`은 전체 기록(이전 모든 출력 및 비평)을 확인하므로 실수를 반복하지 않습니다. 논문에서는 이 요소를 제거했을 때 품질이 급격히 떨어지는 것을 실험적으로 확인했습니다.

주요 결과: GPT-4를 포함한 7개 작업(수학, 코드, 약어, 대화)에서 평균 +20 절대 개선. 추가 학습, 외부 도구 없이 단일 모델로 구현.

### CRITIC (Gou et al., arXiv:2305.11738, v4 2024년 2월)

Self-Refine의 약점: 피드백 단계에서 LLM이 자기 자신을 평가합니다. 사실적 주장에 대해 이는 신뢰할 수 없습니다(환각은 종종 생성한 모델에게 설득력 있게 보입니다). CRITIC은 `feedback(task, output)`을 `verify(task, output, tools)`로 대체하며, `tools`에는 다음이 포함됩니다:

- 사실적 주장을 위한 검색 엔진.
- 코드 정확성을 위한 코드 인터프리터.
- 산술 계산을 위한 계산기.
- 도메인 특화 검증기(유닛 테스트, 타입 체커, 린터).

검증기는 도구 결과에 기반한 구조화된 비평을 생성합니다. 이후 리파이너는 이 비평을 조건으로 합니다.

주요 결과: CRITIC은 비평이 근거에 기반하기 때문에 사실적 작업에서 Self-Refine을 능가합니다. 외부 검증기가 없는 작업(창의적 글쓰기, 포맷팅)에서는 CRITIC이 Self-Refine으로 축소됩니다.

### 중지 조건

두 가지 일반적인 형태:

1. **검증기 통과.** 외부 테스트가 성공을 반환합니다. 사용 가능한 경우 선호됨(유닛 테스트, 타입 체커, 가드레일 어서션).
2. **피드백 미발행.** 모델이 "출력이 적절함"이라고 말합니다. 저렴하지만 신뢰할 수 없음; 최대 반복 횟수와 함께 사용.

2026년 기본값: 둘을 결합. "검증기 통과 시 OR 모델이 '적절함' 판단 AND 반복 횟수 ≥ 2 OR 반복 횟수 ≥ 최대 반복 횟수."

### 평가자-최적화기 (Anthropic, 2024)

Anthropic의 2024년 12월 게시물은 이를 5가지 워크플로우 패턴 중 하나로 명명했습니다. 두 가지 역할:

- 평가자: 출력을 점수화하고 비평을 생성합니다.
- 최적화기: 비평을 기반으로 출력을 수정합니다.

평가자가 통과할 때까지 반복합니다. 이는 Anthropic의 프레임워크에서 Self-Refine/CRITIC에 해당합니다. Anthropic이 추가한 중요한 엔지니어링 세부 사항: 평가자와 최적화기 프롬프트가 구조적으로 달라야 모델이 단순히 승인하지 않습니다.

### OpenAI Agents SDK 출력 가드레일

OpenAI Agents SDK는 이 패턴을 "출력 가드레일"로 제공합니다. 가드레일은 에이전트의 최종 출력에서 실행되는 검증기입니다. 가드레일이 트리거되면(`OutputGuardrailTripwireTriggered` 발생), 출력이 거부되고 에이전트가 재시도할 수 있습니다. 가드레일은 도구(CRITIC 스타일)를 호출하거나 순수 함수(Self-Refine 스타일)일 수 있습니다.

### 2026년 주의 사항

- **고무 도장 루프.** 동일한 모델이 동일한 프롬프트 스타일로 생성과 비평을 수행하면 "문제 없음"으로 수렴합니다. 구조적으로 다른 프롬프트를 사용하거나, 비평을 위해 더 작고 저렴한 모델을 사용하세요.
- **과도한 개선.** 각 리파인 단계는 지연 시간과 토큰을 추가합니다. 1-3회 단계를 예산으로 책정하고, 이후에는 인간 검토로 확대하세요.
- **사소한 작업에 CRITIC 적용.** 외부 검증기가 없는 경우 CRITIC은 Self-Refine으로 퇴화됩니다; 스텁 검증기를 위해 지연 시간을 지불하지 마세요.

## 빌드하기

`code/main.py`는 장난감 작업(주제 주어진 짧은 글머리 기호 목록 생성)에서 Self-Refine과 CRITIC을 구현합니다. 검증기는 형식(3개의 글머리 기호, 각각 60자 이내)을 확인합니다. CRITIC은 알려진 환각(hallucination)을 페널티로 처리하는 외부 "사실 검증기"를 추가합니다.

구성 요소:

- `generate` — 스크립트 기반 생성기.
- `feedback` — LLM 스타일 자체 비평.
- `verify_external` — CRITIC 스타일 근거 기반 검증기.
- `refine` — 이전 기록을 기반으로 출력 재작성.
- 종료 조건 — 검증기 통과 또는 최대 4회 반복.

실행 방법:

```
python3 code/main.py
```

Self-Refine과 CRITIC 실행 결과를 비교하세요. CRITIC은 외부 검증기가 자체 비평가가 갖지 못한 근거를 가지고 있기 때문에 Self-Refine이 놓친 사실적 오류를 잡아냅니다.

## 사용 방법

Anthropic의 평가자-최적화기(evaluator-optimizer)는 Claude 친화적인 언어로 이 패턴을 구현한 것입니다. OpenAI Agents SDK의 출력 가드레일(output guardrails)은 CRITIC 형태(가드레일이 도구를 호출할 수 있음)입니다. LangGraph는 Self-Refine처럼 읽히는 리플렉션(reflection) 노드를 제공합니다. Google의 Gemini 2.5 Computer Use는 CRITIC 변형인 단계별 안전 평가자(per-step safety evaluator)를 추가했습니다: 모든 액션은 커밋 전에 검증됩니다.

## Ship It

`outputs/skill-refine-loop.md`는 작업 형태(task shape), 검증기(verifier) 가용성, 반복 예산(iteration budget)을 기반으로 평가자-최적화기 루프(evaluator-optimizer loop)를 구성합니다. 생성기(generator), 평가자/검증기(evaluator/verifier), 최적화기(optimizer)를 위한 프롬프트와 중단 정책(stop policy)을 출력합니다.

## 연습 문제

1. `max_iterations=1`로 설정하여 토이를 실행해 보세요. CRITIC이 여전히 도움이 되나요?
2. 외부 검증기를 노이즈가 있는 검증기(30% 무작위 오탐)로 교체해 보세요. 루프는 어떤 동작을 하나요? 이는 2026년 대부분의 가드레일 스택의 현실입니다.
3. "다른 모델에서의 생성기-비평기" 변형을 구현해 보세요: 큰 모델이 생성하고, 작은 모델이 비평하는 방식입니다. 동일 모델 대비 성능이 더 좋나요?
4. CRITIC 섹션 3(arXiv:2305.11738 v4)을 읽고, 세 가지 검증 도구 범주의 이름을 각각 하나씩 예시와 함께 작성해 보세요.
5. OpenAI Agents SDK의 `output_guardrails`를 CRITIC의 검증기 역할에 매핑해 보세요. SDK가 잘못 이해한 부분과 올바르게 이해한 부분은 무엇인가요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Self-Refine | "스스로 수정하는 LLM" | 단일 모델 내에서 생성 -> 피드백 -> 개선 루프, 이력 포함 |
| CRITIC | "도구 기반 검증" | 피드백을 외부 검증자(검색, 코드, 계산, 테스트)로 대체 |
| Evaluator-Optimizer | "Anthropic 워크플로우 패턴" | 두 역할 — 평가자가 점수 부여, 최적화기가 수정 — 수렴할 때까지 반복 |
| Output guardrail | "사후 검증" | 에이전트 출력 후 실행되는 OpenAI Agents SDK 검증기 |
| Verify step | "비평 단계" | 핵심 결정 단계: 근거 기반 또는 자체 평가 |
| Refine history | "모델이 이미 시도한 것" | 이전 출력 + 비평이 개선 프롬프트 앞에 추가됨; 생략 시 품질 저하 |
| Rubber-stamp loop | "자기 동의 실패" | 동일 프롬프트 비평에서 "좋아 보임" 반환; 구조적으로 다른 프롬프트로 해결 |
| Stop condition | "수렴 테스트" | 검증자 통과 OR 피드백 없음 AND 반복 횟수 상한; 단일 조건 절대 아님 |

## 추가 자료

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 정론 논문
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 도구 기반 검증
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 평가자-최적화기 워크플로우 패턴
- [OpenAI Agents SDK 문서](https://openai.github.io/openai-agents-python/) — CRITIC 형태의 검증기로서 출력 가드레일