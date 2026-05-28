# Hybrid MFR + ByT5 Plan

목적: MFR을 기본 prediction으로 사용하고, ByT5가 충분히 확신 있는 경우에만 MFR prediction을 덮어쓰는 hybrid system을 validation에서 평가한다.

## 배경

한국어에서 ByT5 단독 prediction은 LAI/MFR보다 낮았다. 특히 changed token을 맞추지 못했고, unchanged token을 과도하게 바꿨다. 따라서 ByT5를 단독 시스템으로 쓰기보다, MFR의 안정성을 유지하면서 ByT5를 보조 후보로 사용하는 전략을 실험한다.

## Hybrid Rule

```text
base_pred = MFR(raw_token)

if ByT5 prediction is safe and margin >= threshold:
    final_pred = ByT5 prediction
else:
    final_pred = base_pred
```

Margin:

```text
margin = raw_nll - byt5_pred_nll
```

margin이 클수록 모델이 raw copy보다 ByT5 prediction을 더 선호한다.

## 스크립트

- `scripts/byt5/hybrid_mfr_byt5_eval.py`

기능:

- train split에서 MFR dictionary 생성
- validation split에서 MFR prediction 생성
- ByT5 checkpoint로 prediction 생성
- raw copy NLL과 ByT5 prediction NLL 계산
- threshold sweep
- hybrid ERR 계산

## 한국어 실행 예시

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/hybrid_mfr_byt5_eval.py \
  --checkpoint outputs/byt5/lang_ko_1epoch_bs4/checkpoint \
  --lang ko \
  --batch-size 32 \
  --thresholds=-2,-1,-0.5,0,0.5,1,2,3,5 \
  --output-dir outputs/byt5/lang_ko_1epoch_bs4/hybrid_mfr_margin_ko
```

출력:

```text
outputs/byt5/lang_ko_1epoch_bs4/hybrid_mfr_margin_ko/hybrid_records.jsonl
outputs/byt5/lang_ko_1epoch_bs4/hybrid_mfr_margin_ko/best_hybrid_predictions.jsonl
outputs/byt5/lang_ko_1epoch_bs4/hybrid_mfr_margin_ko/summary.json
```

## 비교 기준

- LAI ko accuracy: `91.17`
- MFR ko accuracy: `92.18`
- MFR ko ERR: `11.45`
- ByT5 ko 1 epoch accuracy: `80.48`
- ByT5 ko 1 epoch ERR: `-121.08`

확인할 것:

- best hybrid ERR이 MFR보다 높은가
- ByT5 accepted edit 수가 너무 많지 않은가
- changed_correct가 0에서 증가하는가
- unchanged_overedit이 MFR보다 많이 늘어나지 않는가

## 2026-05-29 한국어 실행 결과

```text
Threshold sweep:
threshold=-2.00 acc=80.48 ERR=-121.08 accepted_byt5=1880 changed_correct=0 unchanged_overedit=201
threshold=-1.00 acc=80.48 ERR=-121.08 accepted_byt5=1880 changed_correct=0 unchanged_overedit=201
threshold=-0.50 acc=80.48 ERR=-121.08 accepted_byt5=1880 changed_correct=0 unchanged_overedit=201
threshold=0.00 acc=80.85 ERR=-116.87 accepted_byt5=1871 changed_correct=0 unchanged_overedit=194
threshold=0.50 acc=82.71 ERR=-95.78 accepted_byt5=186 changed_correct=25 unchanged_overedit=184
threshold=1.00 acc=85.32 ERR=-66.27 accepted_byt5=135 changed_correct=25 unchanged_overedit=135
threshold=2.00 acc=91.28 ERR=1.20 accepted_byt5=20 changed_correct=25 unchanged_overedit=23
threshold=3.00 acc=91.86 ERR=7.83 accepted_byt5=8 changed_correct=27 unchanged_overedit=14
threshold=5.00 acc=92.18 ERR=11.45 accepted_byt5=1 changed_correct=27 unchanged_overedit=8
best_threshold: 5.0
best_accuracy: 92.18
best_ERR: 11.45
```

해석:

- best hybrid는 한국어 MFR과 동일한 `accuracy 92.18`, `ERR 11.45`다.
- threshold가 높아질수록 ByT5 output을 거의 거부하고 MFR로 돌아간다.
- 현재 한국어 ByT5 checkpoint는 MFR을 개선하지 못했다.
- 이 결과는 한국어에서는 ByT5 단독/보조 사용보다 MFR이 더 안전하다는 근거다.
- 표의 `changed_correct`는 hybrid 전체 prediction에서 맞춘 changed token 수이며, ByT5가 단독으로 맞춘 수가 아니다. threshold `5.0`에서 `accepted_byt5=1`인데 `changed_correct=27`인 이유는 MFR이 맞춘 changed token이 포함되기 때문이다.
