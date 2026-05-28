# Korean-specific ByT5 Plan

목적: 언어별 ByT5 접근의 첫 실험으로 한국어(`ko`)만 학습하고, 한국어 validation만 평가한다.

한국어를 먼저 보는 이유:

- 과제 guideline에서 Korean이 강조 언어 중 하나다.
- validation split에 `ko`가 포함되어 있다.
- MFR 한국어 validation ERR이 낮았다: `11.45`.
- 언어별 모델이 MFR의 한계를 보완할 수 있는지 빠르게 확인하기 좋다.

## 1. 한국어 train examples 생성

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/build_lang_byt5_examples.py \
  --split train \
  --lang ko \
  --output-dir sample_data/byt5/lang
```

생성 파일:

```text
sample_data/byt5/lang/train_ko_examples.jsonl
sample_data/byt5/lang/train_summary.json
```

## 2. 한국어 ByT5 fine-tuning

처음 실험은 1,000 step으로 시작한다. 학습 중 10 step마다 loss log가 출력된다.

```bash
python scripts/byt5/finetune_byt5.py \
  --examples sample_data/byt5/lang/train_ko_examples.jsonl \
  --max-examples -1 \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --warmup-steps 50 \
  --log-every 10 \
  --output-dir outputs/byt5/lang_ko_1000steps
```

출력:

```text
outputs/byt5/lang_ko_1000steps/run_config.json
outputs/byt5/lang_ko_1000steps/train_log.jsonl
outputs/byt5/lang_ko_1000steps/checkpoint/
```

## 3. 한국어 validation 평가

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/lang_ko_1000steps/checkpoint \
  --lang ko \
  --limit-sentences -1 \
  --batch-size 64 \
  --output-dir outputs/byt5/lang_ko_1000steps/validation_eval_ko
```

출력:

```text
outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl
outputs/byt5/lang_ko_1000steps/validation_eval_ko/summary.json
outputs/byt5/lang_ko_1000steps/validation_eval_ko/summary.md
```

## 4. 비교 기준

한국어 MFR validation 결과:

```text
ko MFR ERR: 11.45
ko LAI Acc: 91.17
ko MFR Acc: 92.18
```

이 실험은 전체 multilingual 성능이 아니라 한국어 language-specific 성능으로 기록한다.

## 2026-05-29 학습 결과

사용자가 한국어 전용 1,000-step fine-tuning을 완료했다.

```text
ByT5 fine-tuning complete
first_loss: 3.240163
last_loss: 0.033255
avg_last_10: 0.549009
log: outputs/byt5/lang_ko_1000steps/train_log.jsonl
checkpoint: outputs/byt5/lang_ko_1000steps/checkpoint
```

해석:

- 학습 loss는 크게 감소했다.
- 이 결과만으로 성능을 판단하지 않는다.
- 다음 단계는 `--lang ko` validation 평가다.

## 2026-05-29 Validation 결과

사용자가 한국어 validation만 평가했다.

```text
Validation evaluation complete
LAI accuracy: 91.17
Model accuracy: 79.41
ERR: -133.13
predictions: outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl
summary: outputs/byt5/lang_ko_1000steps/validation_eval_ko/summary.md
```

해석:

- 이 설정의 한국어 ByT5는 MFR보다 낮다.
- 한국어 MFR ERR `11.45`, MFR accuracy `92.18`과 비교하면 크게 부족하다.
- train loss 감소가 validation 성능 향상으로 이어지지 않았다.
- 다음 분석은 prediction을 확인하여 unchanged token을 과도하게 바꾸는지, 출력이 빈 문자열/반복 문자열로 붕괴하는지, changed token에는 맞는 경우가 있는지 확인해야 한다.

## 2026-05-29 한국어 1 epoch 학습 결과

이전 1000-step 실험은 한국어 전체 데이터를 한 번도 다 보지 못한 설정이었다. 한국어 train token example은 13,130개이므로, batch size 4 기준 1 epoch는 약 3,283 step이다. 이에 따라 3,300 step 학습을 실행했다.

```text
ByT5 fine-tuning complete
first_loss: 3.240163
last_loss: 0.532505
avg_last_10: 0.056538
log: outputs/byt5/lang_ko_1epoch_bs4/train_log.jsonl
checkpoint: outputs/byt5/lang_ko_1epoch_bs4/checkpoint
```

다음 단계:

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/lang_ko_1epoch_bs4/checkpoint \
  --lang ko \
  --limit-sentences -1 \
  --batch-size 64 \
  --output-dir outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko
```

## 2026-05-29 한국어 1 epoch validation 결과

```text
Validation evaluation complete
LAI accuracy: 91.17
Model accuracy: 80.48
ERR: -121.08
predictions: outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/predictions.jsonl
summary: outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/summary.md
```

오류 분석:

```text
total: 1880
buckets: {'unchanged_correct': 1513, 'changed_copied_raw': 157, 'unchanged_overedited': 201, 'changed_wrong_other': 9}
changed accuracy: 0/166
unchanged accuracy: 1513/1714
```

해석:

- 1000-step 실험보다 accuracy는 `79.41 -> 80.48`로 조금 나아졌다.
- 그러나 changed token은 여전히 하나도 맞추지 못했다.
- unchanged over-edit는 `221 -> 201`로 조금 줄었다.
- 이 결과는 단순히 전체 데이터를 1 epoch 보는 것만으로는 한국어 slang/축약 normalization을 배우기 어렵다는 것을 시사한다.
