# Phase 5A Nonnegative Prediction-Support Policy

## 결정 근거

Validation-only 진단에서 raw LightGBM seed 42의 음수 예측은 train 432,324건 중
19건, validation 92,664건 중 2건이었다. Validation 최솟값은
`-9.974868968298846`이었고, 진단용 zero projection 후 MAE 변화는
`-0.000114765484`였다. Project test split에는 접근하지 않았다. 이 관측은 정책의
필요성을 확인하는 진단 근거이며 성능 개선 주장이 아니다.

프로젝트 target의 허용 support는 `y >= 0`이다. 모든 paper-facing 모델에 다음
버전 고정 정책을 동일하게 적용한다.

```text
prediction_support_policy = nonnegative_max_zero_v1
final_prediction = max(0, unprojected_prediction)
```

NaN과 infinity는 projection 전에 거부한다. True target은 변경하지 않는다.

## 모델별 계약

- Identity/original-target 모델은 estimator의 원래 prediction을 진단용으로
  보존하고, final original prediction만 projection한다. Secondary `pred_log`는
  projected original prediction에서 계산한다.
- Log-target 모델은 native `pred_log`를 log-space 평가에 그대로 사용한다.
  Median 또는 Duan inverse로 얻은 original prediction에만 projection한다.
- LightGBM early stopping, neural checkpoint 선택, checkpoint 확정 후 median/Duan
  비교, candidate 선택은 모두 projected original-unit validation MAE를 사용한다.
- 저장 prediction CSV의 `pred_original`은 projected final prediction이다.

모든 신규 metrics와 run manifest는 정책 문자열, projection 전 음수 개수·비율·
최솟값, 실제 projection 개수를 기록한다. 음수 행은 cell ID, 가능한 month ID,
true target, unprojected prediction, final prediction을 별도 validation-only CSV에
기록한다.

## 기존 결과 호환

기존 seed-42 artifact는 수정하지 않는다. 재사용 시 저장된 `pred_original` 전체가
이미 nonnegative인지 읽기 전용으로 검증하고, 통과한 경우에만 in-memory audit
manifest에 `projection_identity_verified=true`를 추가한다. 기존 artifact의 hash나
파일 내용은 바꾸지 않는다.

## LightGBM 모델 artifact

신규 Phase 5A LightGBM run은 fitted booster를 임시 sibling 파일에 쓴 뒤 atomic
replace로 `fitted_model.txt`를 생성한다. 기존 대상 경로는 덮어쓰지 않으며, 실패
시 임시 파일을 정리한다. Model SHA-256과 저장한 best iteration을 manifest에
기록하고, 공식 LightGBM `Booster(model_file=...)` API로 재로드한 validation
prediction이 저장 전 final prediction과 일치하는지 검증한다.

공식 API 근거:

- <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html>
- <https://lightgbm.readthedocs.io/en/stable/Python-Intro.html>

## 범위 제한

이 변경은 data, `label_reg`, persistent split, feature order, 모델 구조 및 Q/K/V를
변경하지 않는다. 실제 project-data full training과 project test 평가는 수행하지
않으며, 새 정책의 검증은 단위·합성 validation-only smoke로 제한한다.
