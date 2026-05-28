# ByT5 Validation Evaluation Plan

목적: ByT5 checkpoint를 validation split에 적용하고, token-level LAI accuracy, model accuracy, ERR을 계산하는 evaluation pipeline을 만든다.

이 스크립트는 tiny checkpoint도 평가할 수 있지만, tiny checkpoint 결과는 성능으로 해석하지 않는다. 먼저 pipeline이 작동하는지 확인하는 용도다.

## 스크립트

- `scripts/byt5/evaluate_checkpoint.py`

기능:

- `data/raw/multilexnorm2026-dev-pub`의 validation parquet 로딩
- 각 validation token을 UFAL-style marked input으로 변환
- checkpoint로 token-level generation 수행
- token prediction과 gold token 비교
- LAI accuracy, model accuracy, ERR 계산
- 언어별 metric 계산
- prediction/summary 저장

## 진행상황 확인

이 스크립트는 `tqdm` progress bar를 사용한다.

출력 예시:

```text
generating:  42%|████▏     | ...
```

앞으로 학습/추론 스크립트에는 진행률 표시와 로그 저장을 기본으로 넣는다.

## 빠른 pipeline 확인

기본값은 validation 첫 50 sentence만 평가한다.

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/evaluate_checkpoint.py
```

기본 출력:

- `outputs/byt5/validation_eval/predictions.jsonl`
- `outputs/byt5/validation_eval/summary.json`
- `outputs/byt5/validation_eval/summary.md`

## 전체 validation 평가

실제 fine-tuned checkpoint가 생긴 뒤 전체 validation을 평가할 때 사용한다.

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/<experiment_name>/checkpoint \
  --limit-sentences -1 \
  --batch-size 16 \
  --output-dir outputs/byt5/<experiment_name>/validation_eval
```

## 특정 언어 validation 평가

언어별 모델을 평가할 때는 `--lang`을 사용한다. 예를 들어 한국어 모델은 한국어 validation sentence만 평가한다.

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/lang_ko_1000steps/checkpoint \
  --lang ko \
  --limit-sentences -1 \
  --batch-size 64 \
  --output-dir outputs/byt5/lang_ko_1000steps/validation_eval_ko
```

## 결과 해석

summary에 기록되는 metric:

- LAI accuracy
- model accuracy
- ERR
- per-language ERR

주의:

- tiny checkpoint의 ERR은 성능으로 기록하지 않는다.
- 실제 fine-tuning 실험에서 나온 checkpoint만 MFR과 비교한다.
- validation에는 `da`, `es`, `it`, `tr`, `trde`가 없으므로 전체 17개 언어 성능처럼 과장하지 않는다.

## 2026-05-28 확인 결과

사용자가 tiny checkpoint로 validation 첫 50문장 pipeline check를 실행했고 정상 완료되었다.

```text
LAI accuracy: 82.62
Model accuracy: 0.00
ERR: -475.29
predictions: outputs/byt5/validation_eval/predictions.jsonl
summary: outputs/byt5/validation_eval/summary.md
```

해석:

- validation inference/evaluation pipeline이 끝까지 실행된다.
- prediction 저장과 summary 저장이 정상 작동한다.
- tiny 20-step checkpoint이므로 이 metric은 성능 결과로 사용하지 않는다.
- 실제 fine-tuning checkpoint가 생기면 같은 script로 validation ERR을 계산할 수 있다.
