# ByT5 Controlled Fine-tuning Plan

목적: smoke test 이후 첫 controlled ByT5 fine-tuning을 수행한다. 이 실험부터는 validation ERR을 계산해 MFR baseline과 비교할 수 있는 후보 checkpoint를 만든다.

## 실험 이름

`context_marked_50k_1000steps`

의미:

- `context_marked`: UFAL-style `<extra_id_0> raw_token <extra_id_1>` 입력
- `50k`: train token examples 50,000개 사용
- `1000steps`: optimizer/training step 1,000회

## 스크립트

- `scripts/byt5/finetune_byt5.py`

특징:

- tqdm progress bar 제공
- 기본 `10` step마다 terminal log 출력
- step별 `loss`, moving average loss, learning rate를 JSONL로 저장
- `run_config.json` 저장
- 최종 checkpoint 저장

## 1. Training examples 생성

프로젝트 루트에서 실행한다.

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/build_byt5_examples.py \
  --max-examples 50000 \
  --output sample_data/byt5/train_examples_50k.jsonl
```

## 2. Fine-tuning 실행

```bash
python scripts/byt5/finetune_byt5.py \
  --examples sample_data/byt5/train_examples_50k.jsonl \
  --max-examples 50000 \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --warmup-steps 50 \
  --log-every 10 \
  --output-dir outputs/byt5/context_marked_50k_1000steps
```

출력:

- `outputs/byt5/context_marked_50k_1000steps/run_config.json`
- `outputs/byt5/context_marked_50k_1000steps/train_log.jsonl`
- `outputs/byt5/context_marked_50k_1000steps/checkpoint/`

## 3. 진행상황 확인

학습 중 10 step마다 다음 형태의 log가 터미널에 출력된다.

```text
step=10 loss=... avg_loss_10=... lr=... elapsed=...m
```

전체 step별 log는 아래 파일에 저장된다.

```text
outputs/byt5/context_marked_50k_1000steps/train_log.jsonl
```

## 4. Validation 평가

학습이 끝난 뒤 전체 validation을 평가한다.

```bash
python scripts/byt5/evaluate_checkpoint.py \
  --checkpoint outputs/byt5/context_marked_50k_1000steps/checkpoint \
  --limit-sentences -1 \
  --batch-size 16 \
  --output-dir outputs/byt5/context_marked_50k_1000steps/validation_eval
```

결과:

- `outputs/byt5/context_marked_50k_1000steps/validation_eval/predictions.jsonl`
- `outputs/byt5/context_marked_50k_1000steps/validation_eval/summary.json`
- `outputs/byt5/context_marked_50k_1000steps/validation_eval/summary.md`

## 5. 해석 기준

이 실험은 첫 ByT5 validation 후보이다. 결과를 해석할 때는 다음과 비교한다.

- MFR validation ERR: `39.02`
- MFR validation accuracy: `92.97`
- LAI validation accuracy: `88.48`

주의:

- validation에는 `da`, `es`, `it`, `tr`, `trde`가 없으므로 17개 언어 전체 성능처럼 쓰지 않는다.
- public leaderboard 결과와 validation 결과를 구분한다.
- loss가 감소해도 ERR이 좋아진다는 보장은 없으므로 validation evaluation을 반드시 실행한다.

## 2026-05-28 실행 결과

학습 결과:

```text
ByT5 fine-tuning complete
first_loss: 13.085028
last_loss: 0.199638
avg_last_10: 0.527551
log: outputs/byt5/context_marked_50k_1000steps/train_log.jsonl
```

Validation 결과:

```text
Validation evaluation complete
LAI accuracy: 88.48
Model accuracy: 70.03
ERR: -160.14
predictions: outputs/byt5/context_marked_50k_1000steps/validation_eval/predictions.jsonl
summary: outputs/byt5/context_marked_50k_1000steps/validation_eval/summary.md
```

해석:

- 학습 loss는 크게 감소했다.
- 그러나 전체 validation ERR은 `-160.14`로 MFR baseline `39.02`보다 훨씬 낮다.
- 이 결과는 ByT5 자체가 부적합하다는 결론이 아니라, 앞부분 50k token sample이 언어 편향되어 있었을 가능성을 보여준다.
- 다음 실험은 언어별 sample 또는 balanced multilingual sample로 설계해야 한다.
