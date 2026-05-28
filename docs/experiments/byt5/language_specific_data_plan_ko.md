# ByT5 Language-specific Data Plan

목적: 언어별 ByT5 모델을 학습할 수 있도록 train split을 언어별 JSONL 파일로 분리한다.

UFAL 2021은 언어/데이터셋별 모델을 따로 학습했다. MultiLexNorm2026도 `lang` column이 있으므로, 언어별 routed system을 만들 수 있다.

## 스크립트

- `scripts/byt5/build_lang_byt5_examples.py`

기능:

- train parquet 읽기
- 언어별로 row filtering
- 각 token을 UFAL-style context-marked example로 변환
- 언어별 JSONL 저장
- 언어별 count summary 저장

## 전체 언어 JSONL 생성

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/build_lang_byt5_examples.py \
  --split train \
  --lang all \
  --output-dir sample_data/byt5/lang
```

출력 예:

```text
sample_data/byt5/lang/train_ko_examples.jsonl
sample_data/byt5/lang/train_de_examples.jsonl
sample_data/byt5/lang/train_vi_examples.jsonl
sample_data/byt5/lang/train_summary.json
```

## 특정 언어만 생성

```bash
python scripts/byt5/build_lang_byt5_examples.py \
  --split train \
  --lang ko \
  --output-dir sample_data/byt5/lang
```

## 언어별 downsampling

언어별로 최대 3,000개만 만들고 싶을 때:

```bash
python scripts/byt5/build_lang_byt5_examples.py \
  --split train \
  --lang all \
  --max-examples-per-lang 3000 \
  --output-dir sample_data/byt5/lang_3k
```

## Changed token만 만들기

실제 normalization이 필요한 token만 학습하고 싶을 때:

```bash
python scripts/byt5/build_lang_byt5_examples.py \
  --split train \
  --lang all \
  --only-changed \
  --output-dir sample_data/byt5/lang_changed
```

주의: `only_changed`는 학습 분포를 바꾸므로 실험 설정에 명확히 기록해야 한다.

## 다음 단계

1. 언어별 example count 확인
2. validation이 있는 언어부터 language-specific fine-tuning 실행
3. 언어별 validation evaluator 옵션 추가
4. multilingual single model과 language-specific model 비교

Validation이 있는 언어:

- `de`, `en`, `hr`, `id`, `iden`, `ja`, `ko`, `nl`, `sl`, `sr`, `th`, `vi`

Validation이 없는 언어:

- `da`, `es`, `it`, `tr`, `trde`

주의: 위 언어들은 official validation split에 없으므로 `evaluate_checkpoint.py --lang da`처럼 실행하면 평가할 validation row가 없다. 이 언어들은 train 내부 holdout/cross-validation을 따로 만들거나, 최종 submission 전에는 MFR 또는 별도 기준으로 판단해야 한다.
