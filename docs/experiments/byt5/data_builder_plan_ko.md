# ByT5 Data Builder Plan

목적: MultiLexNorm2026 parquet 데이터를 UFAL 2021 방식의 ByT5 token-level generation example로 변환한다.

## 배경

MFR은 train에서 본 raw token을 기억해서 가장 빈번한 normalization으로 바꾸는 방식이다. 이 방식은 unseen noisy form에 약하다. ByT5는 byte-level pretrained seq2seq model이므로, raw token과 normalized token 사이의 문자/byte-level 변환 패턴을 학습할 수 있다.

UFAL 2021 시스템은 각 token을 문맥 안에서 표시하고, 해당 token의 normalized form만 출력하도록 ByT5를 fine-tuning했다.

```text
input:  left context <extra_id_0> raw_token <extra_id_1> right context
target: norm_token
```

## 생성한 스크립트

- `scripts/byt5/build_byt5_examples.py`

이 스크립트는 학습을 하지 않는다. parquet 데이터를 읽어서 ByT5 fine-tuning에 사용할 JSONL 예제를 만든다.

출력 column:

- `lang`: language code
- `sentence_id`: 원본 split 안의 sentence index
- `token_id`: sentence 안의 token index
- `raw_token`: 원본 token
- `target_token`: gold normalization token
- `input_text`: ByT5 encoder input
- `target_text`: ByT5 decoder target
- `changed`: `raw_token != target_token`

## 기본 실행

처음에는 전체 데이터를 만들지 않고 200개 sample만 생성한다.

```bash
cd /home/raymond/new_project
conda activate multilexnorm-byt5

python scripts/byt5/build_byt5_examples.py
```

기본 출력:

```text
sample_data/byt5/train_examples_sample.jsonl
```

샘플 확인:

```bash
head -n 5 sample_data/byt5/train_examples_sample.jsonl
```

## Changed token만 샘플링

실제 normalization이 필요한 token만 보고 싶을 때 사용한다.

```bash
python scripts/byt5/build_byt5_examples.py \
  --only-changed \
  --output sample_data/byt5/train_changed_examples_sample.jsonl
```

## 전체 train 예제 생성

모든 token을 학습 예제로 만들려면 `--max-examples -1`을 사용한다.

```bash
python scripts/byt5/build_byt5_examples.py \
  --split train \
  --max-examples -1 \
  --output sample_data/byt5/train_examples.jsonl
```

주의: 모든 token을 예제로 만들면 sentence 수보다 훨씬 큰 데이터가 생성된다. 처음에는 sample로 format을 확인한 뒤 전체 생성을 결정한다.

## 다음 단계

1. sample JSONL이 의도한 format인지 확인
2. ByT5 tokenizer로 `input_text`와 `target_text`가 정상 encode되는지 확인
3. 아주 작은 batch로 forward pass smoke test
4. 이후 fine-tuning script 작성 여부 결정

## 2026-05-28 확인 결과

사용자가 sample builder를 실행했고, 다음과 같은 예제가 생성됨을 확인했다.

```json
{"lang": "da", "sentence_id": 0, "token_id": 3, "raw_token": "tilfaeldigt", "target_token": "tilfældigt", "input_text": "Dette er ikke <extra_id_0> tilfaeldigt <extra_id_1> .", "target_text": "tilfældigt", "changed": true}
```

해석:

- `<extra_id_0>`와 `<extra_id_1>`가 target raw token 주변에 정상 삽입되었다.
- `target_text`는 문장 전체가 아니라 해당 token의 normalized form 하나다.
- 이 형식은 UFAL 2021의 context-marked token-level ByT5 fine-tuning 방식과 일치한다.

다음 작업 후보:

1. ByT5 tokenizer/model 로딩 확인
2. JSONL sample을 읽어 작은 batch로 tokenization 확인
3. forward pass smoke test script 작성
