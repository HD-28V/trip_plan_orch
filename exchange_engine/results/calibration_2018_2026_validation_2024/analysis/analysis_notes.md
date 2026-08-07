# Threshold Calibration 분석 메모

이 문서는 기존 12개 configuration의 사람 검토를 돕는 분석 자료입니다.
production threshold, winner, score 또는 순위를 만들지 않습니다.

## 입력 검증

- configuration 12개
- CALIBRATION / VALIDATION
- USD / JPY / EUR
- horizon 5 / 10 / 20 / 60
- WAIT / WATCH / GOOD / STRONG
- count, ratio, NaN 및 cross-currency 산식 재검증

## 집계 원칙

- 동일 signal occurrence는 네 horizon에 반복되므로 sample count에 한 번만 포함합니다.
- favorable ratio는 favorable count 합계를 evaluable count 합계로 나눕니다.
- mean advantage는 evaluable count로 가중합니다.
- raw outcome이 없으므로 compact median은 evaluable cell median들의 중앙값입니다.
- headline CAL/VAL 값은 모든 evaluable GOOD+STRONG label을 사용합니다.
- 성능 gap은 두 기간에 공통으로 지원되는 통화×horizon cell에서 VALIDATION - CALIBRATION으로 계산하며 signed/absolute 값을 함께 제공합니다.
- 표본 미달 또는 분모 0은 0으로 바꾸지 않고 NaN/UNAVAILABLE로 유지합니다.
- CAL/VAL 기간 길이가 다르므로 raw sample count gap은 occurrence ratio와 함께 봅니다.

## 분석용 heuristic

아래 값은 production 기준이 아니라 저표본과 불안정을 표시하기 위한 공개 heuristic입니다.

- LOW_SAMPLE: evaluable < 25
- VERY_LOW_SAMPLE: evaluable < 10
- LOW_COVERAGE: coverage < 0.80
- SEVERE_OUTCOME_CENSORING: coverage < 0.50
- material favorable gap/range: 0.10
- material mean/median advantage gap: 0.50 pct
- currency favorable range: 0.15
- horizon favorable drop: 0.15
- signal comparison minimum evaluable: 10
- signal comparison minimum comparable cells: 4
- signal comparison failure fraction: 0.50
- positive cell metric votes: 2/3
- positive favorable floor: > 0.50
- positive advantage floor: > 0.00 pct
- positive currency supported-cell ratio: 0.50
- PROMISING positive supported-cell ratio: 0.75
- WEAK core risk-axis count: 2
- WEAK positive supported-cell ratio: < 0.50

### Risk flag 산식

- positive cell: favorable ratio, mean advantage, median advantage 중 설정된 vote 수 이상이 양수 방향입니다.
- CALIBRATION_VALIDATION_SHIFT/DRIFT: favorable·mean·median gap 중 2개 이상이 material 기준을 넘으며, DRIFT는 악화 방향만 셉니다.
- CURRENCY_INSTABILITY: 세 통화가 지원되고 pooled favorable range가 기준 이상이며 mean 또는 median에 부호 충돌이 있습니다.
- HORIZON_INSTABILITY: short/medium evidence가 모두 있고 favorable이 material하게 하락하거나 mean/median이 양수에서 음수로 바뀝니다.
- WEAK_SIGNAL_SEPARATION: 비교 가능한 인접 signal cell에서 WORSE/NOT_SEPARATED 비율이 설정 기준 이상입니다.
- STRONG_NOT_BETTER_THAN_GOOD: 비교 가능한 GOOD→STRONG cell 중 WORSE 비율이 설정 기준 이상입니다.
- STRONG_TOO_RARE: 통화별 STRONG occurrence 최솟값이 VERY_LOW_SAMPLE 기준보다 작습니다.
- 위 flag는 자동 탈락이나 production rule이 아니며 상세 evidence를 찾기 위한 표시입니다.

## 검토 그룹

- PROMISING: 3
- MIXED: 8
- WEAK: 1

PROMISING은 세 통화 모두에 지원되는 긍정 evidence가 있고, 지원 cell의 긍정 비율이 분석용 기준을 충족하며, validation 붕괴나 반복적인 STRONG<GOOD가 없는 검토 그룹입니다.
MIXED는 긍정 evidence와 저표본·불안정·비교 불가가 함께 있는 그룹입니다.
WEAK은 PROMISING 조건을 충족하지 못하면서 여러 독립 risk 축이 반복되거나, 충분한 evidence의 긍정 cell 비율이 낮은 경우입니다.
PROMISING에도 저표본·통화·horizon·separation risk flag는 그대로 남겨 사람 검토에서 함께 봅니다.
표본 부족만으로 WEAK을 부여하지 않습니다.

## 무순위 사람 검토 shortlist

- status: READY_FOR_HUMAN_REVIEW_3_UNRANKED
- count: 3

### baseline__sensitive

- 남은 이유: historical review group=PROMISING; supported currencies=3/3; positive supported cells=10/10; core risk axes=0; human review required
- 위험/제한: LOW_SAMPLE;SEVERE_OUTCOME_CENSORING;HORIZON_EVIDENCE_UNAVAILABLE;STRONG_TOO_RARE

### sma60_sensitive__balanced

- 남은 이유: historical review group=PROMISING; supported currencies=3/3; positive supported cells=10/10; core risk axes=0; human review required
- 위험/제한: VERY_LOW_SAMPLE;SEVERE_OUTCOME_CENSORING;CALIBRATION_VALIDATION_SHIFT;HORIZON_EVIDENCE_UNAVAILABLE;STRONG_TOO_RARE

### sma120_sensitive__balanced

- 남은 이유: historical review group=PROMISING; supported currencies=3/3; positive supported cells=9/10; core risk axes=1; human review required
- 위험/제한: VERY_LOW_SAMPLE;SEVERE_OUTCOME_CENSORING;HORIZON_EVIDENCE_UNAVAILABLE;SIGNAL_SEPARATION_UNAVAILABLE;WEAK_SIGNAL_SEPARATION;STRONG_TOO_RARE

## 해석 제한

- WAIT에는 indicator warm-up과 조건 미충족이 함께 섞여 있을 수 있습니다.
- forward horizon은 서로 중첩되므로 독립 표본이나 독립 실험으로 해석할 수 없습니다.
- 장기 horizon의 tail censoring 때문에 작은 표본의 100% favorable은 강한 증거가 아닙니다.
- 통화·기간별 시장 국면 차이는 threshold 효과와 분리되지 않습니다.
- 이 shortlist 내부에는 우선순위가 없으며 최종 선택은 사람이 수행해야 합니다.
