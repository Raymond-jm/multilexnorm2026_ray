# Korean Conservative Decoding Plan

목적: 한국어 validation에서 `확신이 없으면 raw token을 유지`하는 보수적 decoding을 실험한다.

## 배경

한국어 validation의 LAI accuracy는 `91.17`로 높다. 따라서 모델이 unchanged token을 잘못 바꾸면 큰 손해가 난다. 이전 ByT5 실험은 changed token을 맞추지 못하면서 unchanged token도 과도하게 바꿨다.

보수적 decoding은 다음 원칙을 사용한다.

```text
모델이 prediction을 raw copy보다 충분히 더 선호할 때만 바꾼다.
그렇지 않으면 raw token을 그대로 둔다.
```

## Confidence 정의

각 token에 대해 두 점수를 비교한다.

- `pred_nll`: 모델이 생성한 prediction의 negative log-likelihood
- `raw_nll`: raw token을 그대로 출력하는 negative log-likelihood

정의:

```text
margin = raw_nll - pred_nll
```

해석:

- margin이 클수록 모델이 raw copy보다 generated prediction을 더 선호한다.
- threshold 이상일 때만 model prediction을 채택한다.
- threshold보다 작으면 raw token을 그대로 둔다.

## 스크립트

- `scripts/byt5/evaluate_checkpoint_with_margin.py`

## 실행 예시

한국어 1 epoch checkpoint에 대해 threshold sweep:

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/evaluate_checkpoint_with_margin.py \
  --checkpoint outputs/byt5/lang_ko_1epoch_bs4/checkpoint \
  --lang ko \
  --limit-sentences -1 \
  --batch-size 32 \
  --thresholds "-2,-1,-0.5,0,0.5,1,2,3,5" \
  --output-dir outputs/byt5/lang_ko_1epoch_bs4/conservative_margin_eval_ko
```

출력:

```text
outputs/byt5/lang_ko_1epoch_bs4/conservative_margin_eval_ko/margin_records.jsonl
outputs/byt5/lang_ko_1epoch_bs4/conservative_margin_eval_ko/threshold_sweep.json
```

## 볼 것

- best threshold
- best ERR
- accepted edits 수
- changed correct 수
- unchanged over-edit 수

이 실험의 목표는 ByT5 단독 prediction을 그대로 쓰는 것이 아니라, raw copy fallback을 적용했을 때 LAI/MFR에 가까워질 수 있는지 확인하는 것이다.
