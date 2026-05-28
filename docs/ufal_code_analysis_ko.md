# UFAL MultiLexNorm 2021 코드 분석

Repository:

- https://github.com/ufal/multilexnorm2021

Local path:

- `external/multilexnorm2021-ufal`

확인일: 2026-05-28

## 1. 결론

논문 구현은 공개되어 있으며, ByT5 fine-tuning 방식의 핵심은 코드에서 확인 가능하다. 따라서 원리와 상당 부분의 구조는 따라할 수 있다.

하지만 repo를 MultiLexNorm2026 데이터에 그대로 실행하기는 어렵다. 이유는 다음과 같다.

- 2021 공식 데이터의 `.norm` 파일 형식에 맞춰져 있다.
- 2026 데이터는 Hugging Face parquet 형식이다.
- 2021 언어 config와 wiki dump path가 hard-coded되어 있다.
- synthetic pretraining은 2021 언어별 noise probability와 Wikipedia dump에 의존한다.
- dependency가 오래되었다: `transformers==4.8.2`, `torch==1.9.0+cu111`, `pytorch_lightning==1.3.8`.

따라서 현실적인 접근은 repo를 그대로 실행하기보다, 핵심 아이디어를 2026 데이터 loader에 맞게 이식하는 것이다.

## 2. 핵심 파일

- `train.py`: training loop, Lightning trainer, inference 실행
- `model/model.py`: `T5ForConditionalGeneration.from_pretrained(args.model.pretrained_lm)` 사용
- `data/training_data.py`: train/valid dataset 구성, synthetic pretraining에서 fine-tuning으로 전환
- `data/dataset/multilexnorm.py`: authentic MultiLexNorm data를 token-level generation example로 변환
- `data/dataset/wiki.py`: clean wiki text에 noise를 넣어 synthetic pretraining example 생성
- `data/dataset/augmented.py`: authentic sentence에 on-the-fly noise를 넣는 dataset
- `data/noise/augmenter.py`: typo, accent, casing, repeated letters 등 synthetic noise 구현
- `data/inference_data.py`, `data/dataset/inference.py`: test sentence의 모든 token을 marked-token input으로 변환
- `utility/output_assembler.py`: token별 prediction을 다시 sentence-level output으로 조립

## 3. 실제 fine-tuning input/output

Authentic training data는 `MultilexnormDataset`에서 변환된다.

코드 핵심:

```python
out = self.outputs[sentence_index][word_index]
raw = self.inputs[sentence_index]
raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
raw = " ".join(raw)
return raw, out, sentence_index, word_index
```

즉 입력은 target token을 T5 sentinel token으로 감싼 문장이고, 출력은 해당 token의 gold normalization 하나다.

예시:

```text
input:  Jeg <extra_id_0> skaelver <extra_id_1> .
output: skælver
```

논문에서 설명한 `<X>`, `<Y>`는 실제 구현에서는 `<extra_id_0>`, `<extra_id_1>`이다.

## 4. 모델 구조

`model/model.py`는 Hugging Face `T5ForConditionalGeneration`을 사용한다.

```python
self.model = T5ForConditionalGeneration.from_pretrained(args.model.pretrained_lm)
```

config에서는 다음처럼 지정한다.

```yaml
dataset:
    tokenizer: google/byt5-small

model:
    pretrained_lm: google/byt5-small
```

즉 별도 architecture를 새로 만든 것이 아니라, ByT5 checkpoint를 T5 conditional generation interface로 불러와 fine-tuning한다.

## 5. 학습 단계 구현

논문상의 `synthetic pretraining -> authentic fine-tuning`은 코드에서 `DelayFinetuning` callback으로 구현되어 있다.

`config/en.yaml`:

```yaml
trainer:
    n_epochs: 50
    total_batch_size: 128
    optimizer: adafactor
    lr_decay:
        type: inverse_sqrt
        peak_learning_rate: 0.5e-3
        finetune_learning_rate: 0.1e-3
        warmup_steps: 4000
    delay_finetuning:
        n_steps: 100000

dataset:
    mode: wiki
```

`TrainingData.get_train_dataloader(is_finetuning)`는 `is_finetuning` 값에 따라 dataloader를 바꾼다.

- `is_finetuning == False`: `augmented_train_set`, 즉 wiki/synthetic data
- `is_finetuning == True`: `lexnorm_train_set`, 즉 authentic MultiLexNorm data

`DelayFinetuning`은 training step이 `100000` 이상이 되면 `pl_module.is_finetuning = True`로 바꾸고 dataloader를 reset한다.

따라서 한 번의 training run 안에서 먼저 synthetic data로 학습하고, 이후 authentic data로 fine-tuning하는 구조다.

## 6. Learning rate 구현

`callbacks/lr_decay.py`:

- pretraining 중에는 warmup 후 inverse square root decay
- fine-tuning으로 전환되면 constant fine-tuning LR 사용

설정:

- peak LR: `5e-4`
- warmup: `4000` steps
- fine-tuning LR: `1e-4`

논문 설명과 일치한다.

## 7. Synthetic noise 구현

`data/noise/augmenter.py`에는 여러 noise가 구현되어 있다.

주요 유형:

- typo
- missing apostrophe
- casing change
- accent/diacritic 변형
- missing vowels
- repeated letters
- joined/split words
- 언어별 철자 변형
- Indonesian repetition
- British/American spelling 변환

`WikiDataset`은 clean Wikipedia sentence를 읽고, `Augmenter`로 corrupted sentence와 gold sentence를 만든 뒤, 임의 token 하나를 sentinel로 표시하여 `corrupted context -> clean token` 예제를 만든다.

## 8. Inference 방식

`InferenceDataset`은 test sentence의 모든 token을 각각 하나의 example로 만든다.

```python
raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
```

모델은 각 token의 normalization 후보를 생성한다. 이후 `OutputAssembler`가 sentence id와 word id를 기준으로 token별 예측을 원래 문장 순서로 조립한다.

이 방식은 token-level alignment가 쉬운 대신, 문장 하나를 token 수만큼 모델에 넣어야 해서 inference cost가 크다.

## 9. 2026 과제에 그대로 쓸 수 있는 부분

그대로 참고/이식 가능한 부분:

- marked target token input format
- `T5ForConditionalGeneration` + `google/byt5-small`
- token-level dataset expansion 방식
- token별 prediction assembly 방식
- AdaFactor, LR schedule, fine-tuning hyperparameter 아이디어
- MFR 한계 분석 후 ByT5로 넘어가는 실험 논리

수정이 필요한 부분:

- 2026 Hugging Face parquet loader
- 17개 언어용 language handling
- submission JSON format
- validation/cross-validation split
- 2026 언어에 맞는 noise probability
- Korean/Japanese/Thai/Vietnamese/Indonesian용 synthetic noise 설계
- 최신 PyTorch/Transformers/Lightning 환경 호환성

## 10. 재현 가능성 판단

### 논문 완전 복제

가능하지만 비용이 크다.

- 2021 data와 wiki dump를 맞춰야 한다.
- 100k synthetic pretraining step이 필요하다.
- 언어별 independent model을 학습해야 한다.
- 오래된 dependency를 맞춰야 한다.

### 우리 과제용 현실적 복제

가능하고 추천할 만하다.

먼저 다음만 구현한다.

```text
input:  left context <extra_id_0> raw_token <extra_id_1> right context
output: norm_token
model:  google/byt5-small
```

즉 synthetic pretraining 없이 authentic 2026 train data로 먼저 fine-tuning한다. 이 방식은 UFAL 논문의 핵심 구조를 따르면서 구현 비용을 줄인다.

이후 성능과 시간 여유가 있으면 synthetic noise pretraining을 추가한다.

## 11. 다음 실험 설계 후보

실험 전 사용자 확인 필요.

1. 2026 데이터용 ByT5 marked-token dataset builder 설계
2. MFR 실패 token 중 unseen 비율 분석
3. ByT5-small fine-tuning smoke test를 한 언어 또는 작은 subset에서 실행
4. token-only 입력과 context-marked 입력 비교
5. synthetic pretraining은 나중 단계로 보류
