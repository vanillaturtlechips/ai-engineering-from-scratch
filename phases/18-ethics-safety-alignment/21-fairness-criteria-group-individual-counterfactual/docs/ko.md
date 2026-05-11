# 공정성 기준 — 그룹, 개인, 반사실적

> 공정성 문헌을 구조화하는 세 가지 가족. 그룹 공정성: 인구통계적 평등, 균등 기회, 조건부 사용 정확도 평등 — 보호 그룹 간 평균적 동일률. 개인 공정성(Dwork et al. 2012): 유사한 개인은 유사한 결정을 받음; 결정 맵에 대한 립시츠 조건. 반사실적 공정성(Kusner et al. 2017): 민감한 속성이 반사실적으로 변경되었을 때 변하지 않는 결정이 개인에게 공정함. 2024년 이론적 결과(NeurIPS 2024): CF-정확도 간 본질적 트레이드오프 존재; 모델-불변 방법이 최적이지만 불공정한 예측자를 정확도 손실이 제한된 CF 예측자로 변환. 역추적 반사실(arXiv:2401.13935, 2024년 1월): 법적으로 보호된 속성에 대한 개입을 요구하지 않는 새로운 패러다임. 철학적 조화(ICLR Blogposts 2024): 인과 그래프를 통해 특정 그룹 공정성 측정치를 만족하면 반사실적 공정성이 수반됨.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 세 기준 비교)  
**선수 지식:** 18단계 · 20단계(편향), 02단계(고전적 ML)  
**소요 시간:** ~60분

## 학습 목표

- 세 가지 그룹 공정성 기준(인구통계학적 평등(demographic parity), 균등 기회(equalized odds), 조건부 사용 정확도 평등(conditional use accuracy equality))와 하나의 불가능성 정리(impossibility result)를 설명할 수 있다.
- Dwork et al. 2012의 립시츠(Lipschitz) 공식화를 통해 개인 공정성(individual fairness)을 설명할 수 있다.
- 반사실적 공정성(counterfactual fairness)과 인과 그래프(causal-graph) 의존성을 설명할 수 있다.
- 역추적 반사실(backtracking counterfactuals)과 보호 속성 개입 문제(intervention-on-protected-attribute problem)를 우회하는 이유를 설명할 수 있다.

## 문제 정의

레슨 20은 편향(bias) 측정에 관한 것이었습니다. 레슨 21은 측정이 따라야 할 공정성(fairness) 기준을 정의하는 것에 관한 것입니다. 세 가지 패밀리(family)는 구조적으로 다른 기준을 제공합니다 — 모델이 그룹 공정(group-fair)하지만 개인 불공정(individual-unfair)할 수 있고, 반사실적 공정(counterfactually fair)하지만 그룹 불공정(group-unfair)할 수도 있습니다. 기준 선택은 정책 결정(policy decision) 사항입니다; 어떤 기준도 보편적으로 최적은 아닙니다.

## 개념

### 그룹 공정성

- **인구통계적 평등(Demographic parity).** 모든 그룹에 대해 P(Y=1 | A=a) = P(Y=1 | A=a')를 만족. 동일한 수용률.
- **균등 기회(Equalized odds).** 모든 그룹에 대해 P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')를 만족. 그룹 간 동일한 TPR(True Positive Rate)과 FPR(False Positive Rate).
- **조건부 사용 정확도 평등(Conditional use accuracy equality).** 모든 그룹에 대해 P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')를 만족. 그룹 간 동일한 예측값.

불가능성(Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): 이 세 가지는 불균형한 기저율(unequal base rates) 하에서 동시에 만족될 수 없음.

### 개인 공정성

Dwork et al. 2012. 작업별 유사성 메트릭 d에 대해 결정 맵 f가 |f(x) - f(x')| <= L * d(x, x')를 만족하면(여기서 L은 Lipschitz 상수), f는 개인적으로 공정함. 유사한 개인은 유사한 결정을 받음.

d의 정의가 필요. 통계적 문제가 아닌 정책 문제.

### 반사실적 공정성

Kusner et al. 2017. 인구 집단의 인과 모델에서 개인 i의 민감한 속성을 반사실적으로 변경했을 때 결정이 변하지 않으면, 그 결정은 반사실적으로 공정함.

인과 DAG가 필요. DAG는 모델링 선택 사항. 반사실적 공정성은 DAG의 정당성에 의존.

### 반사실적 공정성-정확도 트레이드오프

NeurIPS 2024 이론: 반사실적 공정성과 예측 정확도 사이에는 본질적인 트레이드오프가 존재. 모델-불문적 방법으로 최적-불공정 예측기를 CF(반사실적 공정성) 예측기로 변환할 수 있으며, 이때 정확도 손실은 제한적. 정확도 손실은 최적 불공정 예측기에서 민감한 속성 계수의 크기에 의존.

### 역추적 반사실

arXiv:2401.13935 (2024년 1월). 전통적 반사실은 민감한 속성에 대한 개입을 요구 — "이 사람이 다른 성별이었다면 결정이 달라졌을까?" 법적 관점에서 이는 문제: 분류법에서 보호 속성은 개입 대상이 될 수 없음.

역추적 반사실은 방향을 전환: 속성 개입이 아닌, 개인의 실제 특성 조합 중 반사실적 결과를 생성했을 조합을 질문. 이는 법적 이의를 회피.

### 철학적 조화

ICLR Blogposts 2024. 인과 그래프를 보유하면 특정 그룹 공정성 측정치를 만족하는 것이 반사실적 공정성을 함의. 세 가지 패밀리는 직교하지 않으며, 동일한 기저 인과 구조의 다른 측면.

이는 불가능성 정리(불균형 기저율로 인한 동시적 그룹 공정성 불가능)를 해결하지는 않음. 그러나 "그룹"과 "개인/반사실" 간 표면적 대립은 인과 모델에 대한 명시적 고려 부재의 인공적 결과임을 보여줌.

### 18단계에서의 위치

레슨 20은 편향 측정. 레슨 21은 공정성 정의. 레슨 22는 프라이버시(차등 프라이버시). 레슨 23은 워터마킹. 이들은 할당-인접 레슨으로, 기만-인접 레슨 7-11을 보완.

## 사용 방법

`code/main.py`는 민감한 속성과 불균형한 기저율을 가진 간단한 이진 분류 데이터셋을 구축합니다. 단순 분류기에 대해 인구통계적 평등(demographic parity), 균등 기회(equalized odds), 조건부 사용 정확도 평등(conditional use accuracy equality)을 계산하고, 세 지표가 서로 일치하지 않는 것을 관찰합니다. 인구통계적 평등을 위한 재가중(re-weighting)을 적용하고 다른 두 지표에 미치는 영향을 확인합니다.

## Ship It

이 레슨은 `outputs/skill-fairness-criterion.md`를 생성합니다. 공정성 주장 또는 정책이 주어졌을 때, 어떤 기준이 주장되고 있는지 식별하고, 모델이 주장된 불평등한 기본 비율 하에서 나머지 기준을 만족할 수 있는지, 그리고 주장이 어떤 인과 DAG에 의존하는지 분석합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 기본 데이터에 대한 세 그룹 메트릭을 보고하세요. 인구통계적-패리티-타겟 재가중(demographic-parity-targeted re-weighting)을 적용한 후 다시 보고하세요.

2. Dwork et al. 2012의 개인 공정성(individual-fairness) 메트릭을 비민감 특성(non-sensitive features)에 L2를 사용하여 구현하세요. Lipschitz 상수 L=1을 위반하는 쌍(pair)이 몇 개인지 보고하세요.

3. Kusner et al. 2017을 읽으세요. 이력서 점수(resume scoring)를 위한 간단한 두 특성 인과 DAG를 구성하고, 이 DAG가 암시하는 반사실적-공정성(counterfactual-fairness) 조건을 식별하세요.

4. 2024년 백트래킹-반사실(backtracking-counterfactuals) 논문은 보호 속성(protected attributes)에 대한 개입(intervention)을 피합니다. 이것이 법적 준수(legal compliance)에 중요한 시나리오를 설명하세요.

5. ICLR 2024 화해(reconciliation) 논문에서는 그룹 공정성(group fairness)과 반사실적 공정성(counterfactual fairness)이 동일한 구조의 측면(facets)이라고 주장합니다. `code/main.py`의 세 기준 중 두 가지를 선택하고, 이 둘을 동등하게 만드는 인과적 가정(causal assumption)을 설명하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 인구통계적 평등(demographic parity) | "동일한 비율(equal rates)" | 그룹 간 P(Y=1 | A=a)가 동일 |
| 균등 기회(equalized odds) | "동일한 TPR/FPR(equal TPR/FPR)" | 그룹 간 동일한 참양성률(true-positive rate)과 거짓양성률(false-positive rate) |
| 조건부 사용 정확도(conditional use accuracy) | "동일한 PPV/NPV(equal PPV/NPV)" | 그룹 간 동일한 예측값(predictive values) |
| 개인 공정성(individual fairness) | "립시츠 조건(Lipschitz condition)" | 유사한 개인에게 유사한 결정 부여 |
| 반사실적 공정성(counterfactual fairness) | "인과적 변경 불변성(causal alteration invariance)" | 반사실적 속성 변경 시 결정 불변 |
| 역추적 반사실(backtracking counterfactual) | "실제 사례를 통한 설명(explain via actuals)" | 속성에서 결과가 아닌 결과로부터 역으로 추론된 반사실 |
| 불가능성 정리(impossibility theorem) | "세 가지 충돌(the three conflict)" | Chouldechova/KMR 2017: 불균등 기저율(unequal base rates) 하에서 그룹 기준 상호배타적 |

## 추가 자료

- [Dwork et al. — 인식을 통한 공정성 (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — 개인 공정성
- [Kusner, Loftus, Russell, Silva — 반사실적 공정성 (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — 반사실적 공정성
- [Chouldechova — 차별적 영향 하의 공정한 예측 (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — 불가능성
- [역추적 반사실 (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — 보호 속성 개입을 위한 새로운 패러다임