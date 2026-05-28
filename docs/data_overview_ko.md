# MultiLexNorm2026 데이터 확인

확인일: 2026-05-28

다운로드 위치:

- `data/raw/multilexnorm2026-dev-pub`
- `data/raw/multilexnorm2026-full-pub`

원본 Hugging Face datasets:

- https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub
- https://huggingface.co/datasets/weerayut/multilexnorm2026-full-pub

## 1. Schema

두 dataset 모두 parquet 형식이며 split은 `train`, `validation`, `test`로 구성된다.

Columns:

- `raw`: list[string], noisy input tokens
- `norm`: list[string], normalized output tokens
- `lang`: string, language code

예시:

```text
raw  = ["Dette", "er", "ikke", "tilfaeldigt", "."]
norm = ["Dette", "er", "ikke", "tilfældigt", "."]
lang = "da"
```

`test` split의 `norm`은 정답이 아니라 token 개수에 맞춘 빈 문자열 리스트다.

```text
raw  = ["Jeg", "skaelver", "."]
norm = ["", "", ""]
lang = "da"
```

## 2. dev-pub와 full-pub 차이

`dev-pub`와 `full-pub`는 `train`과 `validation` split이 완전히 동일하다. 파일 hash도 같다.

다른 부분은 `test` split이다.

| Dataset | Train rows | Validation rows | Test rows |
| --- | ---: | ---: | ---: |
| dev-pub | 39,178 | 8,408 | 5,972 |
| full-pub | 39,178 | 8,408 | 11,956 |

해석:

- `dev-pub`: public dev/test style 확인 및 development용으로 보이는 작은 test split 포함
- `full-pub`: 같은 train/validation에 더 큰 test split 포함
- `full-pub test`는 `dev-pub test`를 포함하지만, row 순서가 단순히 `dev test + extra rows` 형태는 아니다.
- unique `(lang, raw)` key 기준으로는 중복 문장이 있어 dev test row 수 5,972보다 작은 5,914개 key가 full test와 겹친다.

## 3. Train 언어별 row 수

| Lang | Rows |
| --- | ---: |
| da | 719 |
| de | 1,628 |
| en | 2,360 |
| es | 568 |
| hr | 4,760 |
| id | 3,016 |
| iden | 495 |
| it | 593 |
| ja | 2,132 |
| ko | 1,701 |
| nl | 907 |
| sl | 4,670 |
| sr | 4,138 |
| th | 1,750 |
| tr | 570 |
| trde | 800 |
| vi | 8,371 |

총 17개 언어/언어쌍이다.

## 4. Validation 언어별 row 수

Validation에는 12개 언어만 포함되어 있다.

| Lang | Rows |
| --- | ---: |
| de | 573 |
| en | 590 |
| hr | 1,588 |
| id | 431 |
| iden | 165 |
| ja | 305 |
| ko | 212 |
| nl | 308 |
| sl | 1,557 |
| sr | 1,379 |
| th | 250 |
| vi | 1,050 |

Validation에 없는 언어:

- `da`
- `es`
- `it`
- `tr`
- `trde`

이 점 때문에 official validation만으로 17개 언어 전체 generalization을 평가하기 어렵다. 별도 cross-validation 또는 train split 내부 holdout이 필요할 수 있다.

## 5. Test 언어별 row 수

### dev-pub test

| Lang | Rows |
| --- | ---: |
| da | 90 |
| de | 291 |
| en | 983 |
| es | 265 |
| hr | 793 |
| id | 430 |
| iden | 82 |
| it | 50 |
| ja | 304 |
| ko | 107 |
| nl | 154 |
| sl | 778 |
| sr | 688 |
| th | 250 |
| tr | 71 |
| trde | 114 |
| vi | 522 |

### full-pub test

| Lang | Rows |
| --- | ---: |
| da | 181 |
| de | 583 |
| en | 1,967 |
| es | 531 |
| hr | 1,586 |
| id | 861 |
| iden | 165 |
| it | 100 |
| ja | 609 |
| ko | 214 |
| nl | 308 |
| sl | 1,557 |
| sr | 1,377 |
| th | 500 |
| tr | 143 |
| trde | 229 |
| vi | 1,045 |

## 6. Token alignment 확인

현재 확인한 모든 split에서 `len(raw) == len(norm)`이다.

| Dataset | Split | Length mismatch rows |
| --- | --- | ---: |
| dev-pub | train | 0 |
| dev-pub | validation | 0 |
| dev-pub | test | 0 |
| full-pub | train | 0 |
| full-pub | validation | 0 |
| full-pub | test | 0 |

따라서 이 공개 데이터에서는 split/merge correction이 적어도 row-level token length mismatch 형태로 나타나지 않는다. MFR과 token-level seq2seq 접근을 시작하기 쉬운 구조다.

## 7. Normalization 비율

`raw != norm`인 sentence 수:

| Split | Rows | Changed rows |
| --- | ---: | ---: |
| train | 39,178 | 28,134 |
| validation | 8,408 | 5,562 |

`test`는 `norm`이 빈 문자열이므로 모든 row가 `raw != norm`으로 보이며, 실제 normalization 비율로 해석하면 안 된다.

## 8. 실험상 주의점

- `train`과 `validation`은 dev/full에서 동일하므로 모델 학습/검증은 어느 dataset을 읽어도 같다.
- `test` 제출용 예측은 제출 대상이 무엇인지 확인한 뒤 `dev-pub test` 또는 `full-pub test`를 선택해야 한다.
- validation에 5개 언어가 빠져 있으므로, macro-average generalization을 위해 train 내부에서 언어별 holdout/cross-validation을 만드는 방안을 고려해야 한다.
- 단, 새로운 validation/cross-validation 설계를 실험 계획에 넣기 전 사용자 확인이 필요하다.
