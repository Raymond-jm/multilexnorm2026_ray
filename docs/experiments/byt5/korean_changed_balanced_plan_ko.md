# Korean Changed-balanced ByT5 Plan

목적: 한국어 ByT5가 changed token을 하나도 맞추지 못한 문제를 완화하기 위해, changed token signal을 강화한 학습 데이터를 만든다.

## 배경

한국어 train 통계:

```text
train tokens: 13,130
changed tokens: 958
unchanged tokens: 12,172
```

전체 token을 모두 학습하면 unchanged token이 압도적으로 많다. 이전 한국어 1000-step 실험에서는 train loss가 낮아졌지만, validation changed token accuracy가 `0/166`이었다.

이번 실험은 다음 비율을 사용한다.

```text
all changed tokens + 2 * changed_count unchanged tokens
```

한국어 기준 예상:

```text
changed: 958
unchanged sampled: 1,916
total: 2,874
```

## 스크립트

- `scripts/byt5/build_changed_balanced_examples.py`

## 1. Changed-balanced 한국어 데이터 생성

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/build_changed_balanced_examples.py \
  --lang ko \
  --unchanged-ratio 2 \
  --output sample_data/byt5/changed_balanced/train_ko_changed_unchanged2.jsonl
```

출력:

```text
sample_data/byt5/changed_balanced/train_ko_changed_unchanged2.jsonl
sample_data/byt5/changed_balanced/train_ko_changed_unchanged2.summary.json
```

## 2. Fine-tuning

데이터가 작기 때문에 1000 step은 약 1.4 epoch 정도다.

```bash
python scripts/byt5/finetune_byt5.py \
  --examples sample_data/byt5/changed_balanced/train_ko_changed_unchanged2.jsonl \
  --max-examples -1 \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --warmup-steps 50 \
  --log-every 10 \
  --output-dir outputs/byt5/lang_ko_changed_unchanged2_1000steps
```

## 3. 한국어 validation 평가

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/lang_ko_changed_unchanged2_1000steps/checkpoint \
  --lang ko \
  --limit-sentences -1 \
  --batch-size 64 \
  --output-dir outputs/byt5/lang_ko_changed_unchanged2_1000steps/validation_eval_ko
```

## 4. 비교 기준

이전 한국어 ByT5:

```text
accuracy: 79.41
ERR: -133.13
changed accuracy: 0/166
```

한국어 MFR:

```text
accuracy: 92.18
ERR: 11.45
```

확인할 핵심:

- changed token을 하나라도 맞추는가
- unchanged over-edit가 더 심해지는가
- 전체 ERR이 LAI/MFR에 가까워지는가
