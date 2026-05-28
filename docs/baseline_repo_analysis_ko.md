# Baseline GitHub Repository 분석

Repository:

- https://github.com/WeerayutBu/MultiLexNorm2026

Local path:

- `external/MultiLexNorm2026`

확인일: 2026-05-28

## 1. 파일 구조

핵심 파일:

- `README.md`: 설치, 데이터 로딩, MFR baseline, evaluation 예시
- `utils.py`: MFR counting, inference, evaluation, zip 생성 함수
- `demo.ipynb`: 실제 demo notebook
- `requirements.txt`: baseline 실행 dependency
- `outputs/submission_dev.zip`: MFR 예시 제출 파일
- `outputs/submission_full.zip`: MFR 예시 제출 파일

`requirements.txt`:

```text
pyarrow==14.0.2
datasets==2.19.2
numpy<2.0
pandas<2.0
```

현재 로컬에는 `datasets`, `pandas`, `pyarrow`가 없지만 `polars`로 parquet 확인은 가능하다.

## 2. MFR 구현

`utils.py`의 `counting(data)`:

- 각 sentence의 `raw`와 `norm` token을 zip으로 순회한다.
- `counts[wordRaw][wordGold] += 1` 형태로 raw token별 gold token 빈도를 저장한다.

`mfr(input_sent, counts)`:

- input sentence의 각 token을 확인한다.
- token이 dictionary에 있으면 가장 빈번한 replacement를 선택한다.
- 없으면 원래 token을 그대로 둔다.

즉 MFR은 token-level dictionary replacement이다.

## 3. Evaluation 구현

`evaluate(raw, gold, pred)`는 token-level accuracy와 ERR을 계산한다.

계산 방식:

- `changed`: `raw != gold`인 token 수
- `cor`: `gold == pred`인 token 수
- `total`: 전체 token 수
- `accuracy = cor / total`
- `lai = (total - changed) / total`
- `err = (accuracy - lai) / (1 - lai)`

출력 예시:

```text
Baseline acc.(LAI): 93.10
Accuracy:           97.37
ERR:                61.93
```

## 4. Notebook 흐름

`demo.ipynb`에는 두 가지 흐름이 있다.

### 4.1 Validation 평가 예시

```python
dev_pub_data = load_dataset("weerayut/multilexnorm2026-dev-pub")
train, val = dev_pub_data["train"], dev_pub_data["validation"]

counts = counting(train)
ds = pd.DataFrame(val)
ds["pred"] = ds["raw"].apply(lambda x: mfr(x, counts))
evaluate(ds["raw"].tolist(), ds["norm"].tolist(), ds["pred"].tolist())
```

이 예시는 전체 train을 하나의 multilingual dictionary로 사용한다. 언어별 dictionary가 아니다.

### 4.2 Submission 생성 예시

```python
data, save_path = load_dataset("weerayut/multilexnorm2026-dev-pub"), "outputs/submission_dev"
# data, save_path = load_dataset("weerayut/multilexnorm2026-full-pub"), "outputs/submission_full"

train = concatenate_datasets([data["train"], data["validation"]])
out = prediction(train, data["test"])
out.to_json(f"{save_path}/predictions.json", orient="records")
zip_files_flat(save_path, f"{save_path}.zip")
```

`prediction(train, test)`는 언어별 dictionary를 만든다.

```python
for lang in train_df["lang"].unique():
    train_lang = train_df.loc[train_df["lang"] == lang]
    count_langs[lang] = counting(train_lang.to_dict(orient="records"))
```

그 뒤 test row의 `lang`에 맞는 dictionary로 `mfr`를 적용한다.

즉 submission 생성은 `train + validation`을 사용한 language-routed MFR이다.

## 5. 제출 파일 format

`predictions.json`은 JSON list이며 각 item은 다음 필드를 포함한다.

- `raw`: list[string]
- `norm`: list[string], test에서는 빈 문자열 리스트
- `lang`: string
- `pred`: list[string]

예시:

```json
{
  "raw": ["Jeg", "skaelver", "."],
  "norm": ["", "", ""],
  "lang": "da",
  "pred": ["Jeg", "skaelver", "."]
}
```

제출 zip은 flat structure로 `predictions.json`만 포함하면 되는 것으로 보인다.

## 6. 주의점

- README의 dataset 이름 예시 중 `weerayut/multilexnorm2026-pub`는 현재 사용자가 준 링크와 다르다. 실제로는 `dev-pub`와 `full-pub`를 사용한다.
- README의 final phase 링크 텍스트가 `full-pub`가 아니라 `dev-pub`로 반복되어 있다. 실제 데이터셋은 `weerayut/multilexnorm2026-full-pub`가 존재한다.
- validation 평가 예시는 multilingual single dictionary이고, submission 생성 예시는 per-language dictionary이다. 보고서에는 어떤 설정을 사용했는지 명확히 적어야 한다.
- validation split에는 5개 언어(`da`, `es`, `it`, `tr`, `trde`)가 없으므로 validation ERR만으로 모든 언어의 성능을 대표하기 어렵다.
- baseline MFR은 raw token 표면형만 사용하므로 unseen token, typo generalization, context-dependent normalization에 약하다.

## 7. 다음 선택지

실험 실행 전 사용자 확인 필요.

1. baseline repo의 MFR을 그대로 실행하여 validation ERR 재현
2. polars 기반으로 dependency 없이 MFR을 직접 구현하고 validation ERR 계산
3. language-routed MFR과 multilingual single MFR을 validation에서 비교
4. train 내부 holdout/cross-validation 설계를 먼저 논의
