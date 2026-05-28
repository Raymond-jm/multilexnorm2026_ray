# Experiment Log

아직 로컬 실험은 실행하지 않았다.

## 2026-05-28

- Literature review only.
- 읽은 논문:
  - Samuel and Straka (2021), "UFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5"
  - van der Goot et al. (2021), "MultiLexNorm: A Shared Task on Multilingual Lexical Normalization"
- 실험 결과: TBD

## 2026-05-28 Data Inspection

- Downloaded Hugging Face datasets:
  - `weerayut/multilexnorm2026-dev-pub`
  - `weerayut/multilexnorm2026-full-pub`
- Local paths:
  - `data/raw/multilexnorm2026-dev-pub`
  - `data/raw/multilexnorm2026-full-pub`
- No model experiment was run.
- Summary:
  - Schema: `raw: list[string]`, `norm: list[string]`, `lang: string`
  - `train`: 39,178 rows
  - `validation`: 8,408 rows
  - `dev-pub test`: 5,972 rows
  - `full-pub test`: 11,956 rows
  - `train` and `validation` files are identical between dev/full.
  - Only `test` differs.

## 2026-05-28 MFR Baseline Validation

- 실행자: user
- 실행 위치: `external/MultiLexNorm2026`
- Dataset: `weerayut/multilexnorm2026-dev-pub`
- Train split: `train`
- Validation split: `validation`
- Method: baseline repository `utils.counting`, `utils.mfr`, `utils.evaluate`
- Dictionary setting: multilingual single MFR dictionary over all train examples
- Command summary:

```bash
python - <<'PY'
import pandas as pd
from datasets import load_dataset
from utils import counting, mfr, evaluate

data = load_dataset("weerayut/multilexnorm2026-dev-pub")
train = data["train"]
val = data["validation"]

counts = counting(train)

ds = pd.DataFrame(val)
ds["pred"] = ds["raw"].apply(lambda x: mfr(x, counts))

evaluate(
    raw=ds["raw"].tolist(),
    gold=ds["norm"].tolist(),
    pred=ds["pred"].tolist()
)
PY
```

- Result:

| Metric | Value |
| --- | ---: |
| Baseline acc. (LAI) | 88.48 |
| Accuracy | 92.97 |
| ERR | 39.02 |

- Notes:
  - This reproduces the provided MFR-style validation baseline.
  - Next analysis target: identify how many MFR errors come from unseen raw tokens, to motivate ByT5 fine-tuning.

## 2026-05-28 Literature: Pretrained Models

- No model experiment was run.
- 추가로 확인한 문헌:
  - Xue et al. (2022), "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
  - van der Goot (2019), "MoNoise: A Multi-lingual and Easy-to-use Lexical Normalization Tool"
  - van der Goot (2021), "CL-MoNoise: Cross-lingual Lexical Normalization"
  - Bucur and Dinu (2021), "Sequence-to-Sequence Lexical Normalization with Multilingual Transformers"
  - Lourentzou et al. (2019), "Adapting Sequence to Sequence Models for Text Normalization in Social Media"

## 2026-05-28 ByT5 Data Builder Smoke Test

- 실행자: user
- Script: `scripts/byt5/build_byt5_examples.py`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: `train`
- Output: `sample_data/byt5/train_examples_sample.jsonl`
- Purpose: Convert MultiLexNorm2026 parquet rows into UFAL-style marked-token ByT5 examples.
- Result: sample JSONL format is correct.

Example output:

```json
{"lang": "da", "sentence_id": 0, "token_id": 3, "raw_token": "tilfaeldigt", "target_token": "tilfældigt", "input_text": "Dette er ikke <extra_id_0> tilfaeldigt <extra_id_1> .", "target_text": "tilfældigt", "changed": true}
```

- Notes:
  - The target token is marked with `<extra_id_0>` and `<extra_id_1>`.
  - The decoder target is only the normalized token, not the whole sentence.
  - This matches the core UFAL 2021 token-level ByT5 fine-tuning format.

## 2026-05-28 ByT5 Tokenization/Forward Smoke Test

- 실행자: user
- Script: `scripts/byt5/smoke_test_byt5.py`
- Input examples: `sample_data/byt5/train_examples_sample.jsonl`
- Model: `google/byt5-small`
- Purpose: Check tokenizer/model loading and forward loss on UFAL-style marked-token examples.
- Result: passed.

Output:

```text
input_ids shape: (4, 32)
labels shape: (4, 12)
loss: 9.298801
untrained generation for first input: er dette o er ikke tilfaeldigt
ByT5 smoke test passed
```

- Notes:
  - No fine-tuning or parameter update was performed.
  - The finite loss confirms that the current ByT5 input/target format is compatible with `google/byt5-small`.
  - The generation is from an unfine-tuned model, so quality is not interpreted as task performance.

## 2026-05-28 ByT5 Tiny Fine-tuning Smoke Test

- 실행자: user
- Script: `scripts/byt5/tiny_finetune_byt5.py`
- Input examples: `sample_data/byt5/train_examples_sample.jsonl`
- Model: `google/byt5-small`
- Purpose: Check training loop, optimizer update, logging, and checkpoint saving on a tiny sample.
- Setting:
  - examples: 200
  - max steps: 20
  - batch size: 2
  - learning rate: 5e-5
- Result: completed.

Output:

```text
Tiny fine-tuning complete
first_loss: 9.629842
last_loss: 10.530208
log: outputs/byt5/tiny_finetune/train_log.jsonl
checkpoint: outputs/byt5/tiny_finetune/checkpoint
```

- Notes:
  - This is not a performance experiment.
  - Loss did not decrease in this 20-step tiny run, but the purpose was pipeline validation.
  - Checkpoint saving works.
  - Do not compare this result with MFR ERR.

## 2026-05-28 ByT5 Tiny Checkpoint Generation Smoke Test

- 실행자: user
- Script: `scripts/byt5/test_checkpoint_generation.py`
- Checkpoint: `outputs/byt5/tiny_finetune/checkpoint`
- Input examples: `sample_data/byt5/train_examples_sample.jsonl`
- Output: `outputs/byt5/tiny_finetune/generation_sample.jsonl`
- Purpose: Check that the tiny fine-tuned checkpoint can be loaded and used for generation.
- Result: completed.

Output excerpt:

```text
examples: 10
output: outputs/byt5/tiny_finetune/generation_sample.jsonl
First predictions:
- raw='Dette' target='Dette' pred='er ikke tilfaeldigt . . . . . .' changed=False
- raw='er' target='er' pred=' ikke tilfaeldigt ...\nDette K e' changed=False
- raw='ikke' target='ikke' pred=' tilfaeldigt . . . . . . . . . ' changed=False
- raw='tilfaeldigt' target='tilfældigt' pred=' . . . . . . . . . . . . . . . ' changed=True
- raw='.' target='.' pred=' . . . . . . . . . . . . . . . ' changed=False
```

- Notes:
  - This is not a performance result.
  - Generation quality is poor because the checkpoint is from a tiny 20-step smoke test.
  - The important result is that checkpoint loading and generation work.

## 2026-05-28 ByT5 Validation Evaluation Pipeline Smoke Test

- 실행자: user
- Script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/tiny_finetune/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: validation
- Limit: first 50 validation sentences
- Output:
  - `outputs/byt5/validation_eval/predictions.jsonl`
  - `outputs/byt5/validation_eval/summary.md`
- Purpose: Verify that checkpoint inference and ERR computation pipeline works.
- Result: completed.

Output:

```text
LAI accuracy: 82.62
Model accuracy: 0.00
ERR: -475.29
predictions: outputs/byt5/validation_eval/predictions.jsonl
summary: outputs/byt5/validation_eval/summary.md
```

- Notes:
  - This is not a model performance result.
  - The checkpoint is from a 20-step tiny smoke test, so poor accuracy is expected.
  - The important result is that validation tokenization, generation, prediction saving, and ERR computation all complete.

## 2026-05-28 ByT5 5k/100-step Smoke Fine-tuning

- 실행자: user
- Data builder:

```bash
python scripts/byt5/build_byt5_examples.py \
  --max-examples 5000 \
  --output sample_data/byt5/train_examples_5k.jsonl
```

- Fine-tuning command:

```bash
python scripts/byt5/tiny_finetune_byt5.py \
  --examples sample_data/byt5/train_examples_5k.jsonl \
  --max-examples 5000 \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --output-dir outputs/byt5/smoke_5k_100steps
```

- Model: `google/byt5-small`
- Purpose: longer smoke test to check whether training loss can decrease.
- Result: completed.

Output:

```text
Tiny fine-tuning complete
first_loss: 14.825723
last_loss: 3.635378
log: outputs/byt5/smoke_5k_100steps/train_log.jsonl
checkpoint: outputs/byt5/smoke_5k_100steps/checkpoint
```

- Notes:
  - This is still a smoke test, not a final performance experiment.
  - Loss decreased substantially from `14.825723` to `3.635378`.
  - The result supports moving to a controlled validation experiment.

## 2026-05-28 ByT5 Context-marked 50k/1000-step Training

- 실행자: user
- Training examples: `sample_data/byt5/train_examples_50k.jsonl`
- Model: `google/byt5-small`
- Input format: UFAL-style context-marked token input
- Output directory: `outputs/byt5/context_marked_50k_1000steps`
- Setting:
  - max examples: 50,000
  - max steps: 1,000
  - batch size: 4
  - learning rate: 5e-5
  - warmup steps: 50
  - log every: 10 steps
- Training result:

```text
ByT5 fine-tuning complete
first_loss: 13.085028
last_loss: 0.199638
avg_last_10: 0.527551
log: outputs/byt5/context_marked_50k_1000steps/train_log.jsonl
```

- Notes:
  - Loss decreased strongly.
  - The training examples were the first 50k token examples from train, so the sample is likely language-biased.
  - This run should not be treated as the final multilingual ByT5 result.

## 2026-05-28 ByT5 Context-marked 50k/1000-step Validation

- 실행자: user
- Evaluation script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/context_marked_50k_1000steps/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: full validation
- Output:
  - `outputs/byt5/context_marked_50k_1000steps/validation_eval/predictions.jsonl`
  - `outputs/byt5/context_marked_50k_1000steps/validation_eval/summary.md`
- Result:

```text
Validation evaluation complete
LAI accuracy: 88.48
Model accuracy: 70.03
ERR: -160.14
predictions: outputs/byt5/context_marked_50k_1000steps/validation_eval/predictions.jsonl
summary: outputs/byt5/context_marked_50k_1000steps/validation_eval/summary.md
```

- Interpretation:
  - This model is far below the MFR validation baseline (`ERR 39.02`).
  - Because `train_examples_50k.jsonl` was generated from the beginning of train, it is likely language-biased.
  - The result motivates language-balanced or language-specific training before drawing conclusions about ByT5.

## 2026-05-29 ByT5 Korean-specific 1000-step Training

- 실행자: user
- Training examples: `sample_data/byt5/lang/train_ko_examples.jsonl`
- Model: `google/byt5-small`
- Language: `ko`
- Input format: UFAL-style context-marked token input
- Output directory: `outputs/byt5/lang_ko_1000steps`
- Setting:
  - max examples: all Korean train token examples
  - max steps: 1,000
  - batch size: 4
  - learning rate: 5e-5
  - warmup steps: 50
  - log every: 10 steps
- Training result:

```text
ByT5 fine-tuning complete
first_loss: 3.240163
last_loss: 0.033255
avg_last_10: 0.549009
log: outputs/byt5/lang_ko_1000steps/train_log.jsonl
checkpoint: outputs/byt5/lang_ko_1000steps/checkpoint
```

- Notes:
  - This is a language-specific Korean model.
  - Validation must be run with `--lang ko`.
  - Compare against Korean MFR validation ERR `11.45`, not global MFR ERR.

## 2026-05-29 ByT5 Korean-specific 1000-step Validation

- 실행자: user
- Evaluation script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/lang_ko_1000steps/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: validation
- Language filter: `ko`
- Output:
  - `outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl`
  - `outputs/byt5/lang_ko_1000steps/validation_eval_ko/summary.md`
- Result:

```text
Validation evaluation complete
LAI accuracy: 91.17
Model accuracy: 79.41
ERR: -133.13
predictions: outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl
summary: outputs/byt5/lang_ko_1000steps/validation_eval_ko/summary.md
```

- Comparison:
  - Korean MFR validation ERR: `11.45`
  - Korean MFR validation accuracy: `92.18`
  - Korean LAI validation accuracy: `91.17`
- Interpretation:
  - Korean-specific ByT5 is far below MFR and LAI in this setting.
  - The low train loss does not translate to validation performance.
  - Likely issues to inspect: over-editing unchanged tokens, too few effective epochs, output format collapse, or generation copying failure.

## 2026-05-29 ByT5 Korean-specific Error Analysis

- 분석 대상: `outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl`
- Total validation tokens: 1,880
- Buckets:

| Bucket | Count | Percent |
| --- | ---: | ---: |
| unchanged_correct | 1,493 | 79.41 |
| unchanged_overedited | 221 | 11.76 |
| changed_copied_raw | 156 | 8.30 |
| changed_wrong_other | 10 | 0.53 |

- Changed token accuracy: `0 / 166 = 0.00%`
- Unchanged token accuracy: `1493 / 1714 = 87.11%`
- Interpretation:
  - The model did not correctly normalize any changed Korean validation token.
  - Most changed-token failures copied the raw token.
  - It also over-edited 221 unchanged tokens.
  - This explains the negative ERR despite low training loss.
- Detailed note: `docs/experiments/byt5/korean_error_analysis_ko.md`

## 2026-05-29 Korean Hybrid MFR + ByT5 Margin Evaluation

- 실행자: user
- Script: `scripts/byt5/hybrid_mfr_byt5_eval.py`
- Checkpoint: `outputs/byt5/lang_ko_1epoch_bs4/checkpoint`
- Language filter: `ko`
- Strategy: MFR default, ByT5 override only when safety filter passes and margin exceeds threshold.
- Result:

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

- Interpretation:
  - The best hybrid result equals Korean MFR (`accuracy 92.18`, `ERR 11.45`).
  - ByT5 does not improve Korean validation in this checkpoint.
  - Higher thresholds mostly reject ByT5 outputs and recover MFR behavior.
  - `changed_correct` here counts all correct changed tokens in the hybrid output, including MFR corrections; it is not the number of ByT5-only correct edits.

## 2026-05-29 ByT5 Korean-specific 1-epoch Training

- 실행자: user
- Training examples: `sample_data/byt5/lang/train_ko_examples.jsonl`
- Model: `google/byt5-small`
- Language: `ko`
- Input format: UFAL-style context-marked token input
- Output directory: `outputs/byt5/lang_ko_1epoch_bs4`
- Setting:
  - max examples: all Korean train token examples (`13,130`)
  - max steps: 3,300
  - batch size: 4
  - learning rate: 5e-5
  - warmup steps: 100
  - log every: 10 steps
- Training result:

```text
ByT5 fine-tuning complete
first_loss: 3.240163
last_loss: 0.532505
avg_last_10: 0.056538
log: outputs/byt5/lang_ko_1epoch_bs4/train_log.jsonl
checkpoint: outputs/byt5/lang_ko_1epoch_bs4/checkpoint
```

- Notes:
  - This is a fairer Korean-specific setting than the previous 1000-step run because it approximately covers the full Korean train token set once.
  - Validation must be run with `--lang ko`.

## 2026-05-29 ByT5 Korean-specific 1-epoch Validation

- 실행자: user
- Evaluation script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/lang_ko_1epoch_bs4/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: validation
- Language filter: `ko`
- Output:
  - `outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/predictions.jsonl`
  - `outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/summary.md`
- Result:

```text
Validation evaluation complete
LAI accuracy: 91.17
Model accuracy: 80.48
ERR: -121.08
predictions: outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/predictions.jsonl
summary: outputs/byt5/lang_ko_1epoch_bs4/validation_eval_ko/summary.md
```

- Error analysis:

```text
total: 1880
buckets: {'unchanged_correct': 1513, 'changed_copied_raw': 157, 'unchanged_overedited': 201, 'changed_wrong_other': 9}
changed accuracy: 0/166
unchanged accuracy: 1513/1714
```

- Interpretation:
  - 1 epoch improves accuracy slightly over the 1000-step run (`79.41 -> 80.48`) but remains far below MFR.
  - The model still does not correctly normalize any changed Korean validation token.
  - More exposure to all tokens reduced over-editing slightly (`221 -> 201`) but did not solve changed-token normalization.

## 2026-05-29 ByT5 German-specific 1-epoch Validation

- 실행자: user
- Evaluation script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/lang_de_1epoch_bs4/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: validation
- Language filter: `de`
- Output:
  - `outputs/byt5/lang_de_1epoch_bs4/validation_eval_de/predictions.jsonl`
  - `outputs/byt5/lang_de_1epoch_bs4/validation_eval_de/summary.md`
- Result:

```text
Validation evaluation complete
LAI accuracy: 82.04
Model accuracy: 86.60
ERR: 25.43
predictions: outputs/byt5/lang_de_1epoch_bs4/validation_eval_de/predictions.jsonl
summary: outputs/byt5/lang_de_1epoch_bs4/validation_eval_de/summary.md
```

- Interpretation:
  - Unlike Korean, German-specific ByT5 improves over LAI (`82.04 -> 86.60`) with positive ERR (`25.43`).
  - This suggests that the same UFAL-style ByT5 setup can be useful for some languages, but its benefit is language-dependent.
  - Next comparison target: German MFR and German error analysis, to check whether ByT5 is improving over dictionary-based replacement or only over LAI.

## 2026-05-29 ByT5 English-specific 1-epoch Validation

- 실행자: user
- Evaluation script: `scripts/byt5/evaluate_checkpoint.py`
- Checkpoint: `outputs/byt5/lang_en_1epoch_bs4/checkpoint`
- Dataset: `data/raw/multilexnorm2026-dev-pub`
- Split: validation
- Language filter: `en`
- Output:
  - `outputs/byt5/lang_en_1epoch_bs4/validation_eval_en/predictions.jsonl`
  - `outputs/byt5/lang_en_1epoch_bs4/validation_eval_en/summary.md`
- Training setting:
  - Training examples: `35,216`
  - Max steps: `8,804`
  - Batch size: `4`
  - Learning rate: `5e-5`
  - Warmup steps: `100`
  - Log every: `10`
- Validation result:

```text
LAI accuracy: 93.10
MFR accuracy: 95.09
MFR ERR: 28.91
ByT5 accuracy: 96.14
ByT5 ERR: 44.08
Delta ERR (ByT5 - MFR): +15.17
```

- Example predictions from the validation summary:
  - `bruh -> brother` was correctly normalized by ByT5.
  - `yo -> your` was predicted as `you`, showing that slang/abbreviation expansion can still be ambiguous.
  - `dese -> these` was copied as `dese`, showing remaining unseen/noisy-form failures.
- Interpretation:
  - English-specific ByT5 clearly improves over both LAI and global MFR in this 1-epoch setting.
  - Together with German, this supports the hypothesis that byte-level seq2seq fine-tuning can improve over dictionary replacement for some languages.
  - The contrast with Korean indicates that performance is strongly language- and data-dependent.

## 2026-05-29 Korean ByT5 Failure Analysis

- Purpose: explain why the UFAL-style token-level ByT5 setup failed on Korean while improving German and English.
- Compared languages: `ko`, `en`, `de`
- Main observations:

| Metric | ko | en | de |
| --- | ---: | ---: | ---: |
| Train changed tokens | 958 | 2,666 | 2,578 |
| Validation changed tokens | 166 | 633 | 873 |
| Val changed raw seen in global MFR dictionary | 36.1% | 80.1% | 67.9% |
| Val changed exact pair seen in same-language train changed pairs | 26.5% | 73.5% | 49.7% |
| Global MFR correct on changed validation tokens | 16.3% | 49.8% | 27.3% |
| ByT5 changed-token accuracy | 0 / 166 = 0.0% | 295 / 633 = 46.6% | 271 / 873 = 31.0% |
| Train examples over `max_input_length=256` | 13.2% | 0.0% | 0.0% |
| Changed train examples over `max_input_length=256` | 5.2% | 0.0% | 0.0% |
| Changed targets over `max_target_length=32` | 0.9% | 0.0% | 0.0% |

- Korean changed-pair examples:
  - Frequent train pairs include `존나 -> 매우`, `ㄹㅇ -> 진짜`, `시발 -> 이런`, `ㅅㅂ -> 이런`, `걍 -> 그냥`.
  - Validation includes sparse and morphologically attached forms such as `여친이랑 -> 여자친구랑`, `띵작이다 -> 명작이다`, `개씹존잘 -> 매우 잘생긴 사람`, `개상디언이 -> 경상도인들이`.
- Interpretation:
  - Korean has much fewer changed-token examples than English/German.
  - Korean validation changed tokens are less likely to repeat raw forms or exact mappings seen in training.
  - Many Korean changes are slang/abbreviation/profanity replacement or phrase-level semantic normalization, not only spelling repair.
  - Korean particles/endings are often attached to the same eojeol token, so the model must learn both internal lexical replacement and suffix preservation in one output token.
  - UTF-8 byte-level input is longer for Korean, and the current `max_input_length=256` truncates Korean examples more often than English/German.
  - These factors make direct 1-epoch ByT5 fine-tuning unstable for Korean and explain why MFR remains the safer Korean baseline so far.
