# MultiLexNorm2026 보고서 한국어 초안

> 작성 원칙: 이 문서는 최종 ACL-style 영어 보고서로 옮기기 전의 한국어 working draft이다. 실제 실험 결과가 없는 수치, 설정, 데이터 split, 모델 hyperparameter는 추측하지 않고 `TBD`로 둔다.

## 제목 후보

TBD

후보:

- Dictionary Baselines and Byte-level Sequence-to-Sequence Modeling for MultiLexNorm2026
- A Reproducible Baseline Study for Multilingual Lexical Normalization
- From MFR to Byte-level Generation: A Study on MultiLexNorm2026

## Abstract

본 보고서는 MultiLexNorm2026 multilingual lexical normalization 과제에 대한 실험 과정을 다룬다. Lexical normalization은 SNS와 온라인 커뮤니티에서 나타나는 비표준 어휘를 표준형으로 변환하는 작업이며, 본 과제의 공식 평가는 언어별 Error Reduction Rate(ERR)의 macro-average를 중심으로 이루어진다. 우리는 먼저 제공 baseline인 Most-Frequent-Replacement(MFR)를 재현하고, 언어별로 어떤 유형의 normalization이 dictionary 기반 접근으로 해결되는지 분석한다. 이후 TBD 접근을 비교하여 noisy multilingual text에서 byte-level 또는 sequence-to-sequence 모델이 제공하는 일반화 효과를 검토한다. 실험 결과, TBD. 본 보고서는 leaderboard 점수뿐 아니라 validation/cross-validation 기반 일반화 성능과 재현 가능성을 함께 기록한다.

## 1. Introduction

SNS와 온라인 커뮤니티의 텍스트는 표준 문어체와 다른 형태를 자주 포함한다. 예를 들어 축약어, 반복 문자, 오탈자, 비표준 철자, code-switching, 대소문자 변형 등이 나타난다. 이러한 표현은 사람에게는 자연스럽게 이해되지만, dependency parsing, POS tagging, information extraction 등 downstream NLP 시스템에는 오류를 유발할 수 있다.

Lexical normalization은 이러한 noisy text의 비표준 어휘를 canonical form으로 바꾸는 작업이다. MultiLexNorm2026 과제는 17개 언어를 대상으로 multilingual lexical normalization 시스템을 평가한다. 공식 지표는 ERR(Error Reduction Rate)이며, 이는 아무 것도 바꾸지 않는 leave-as-is baseline 대비 word-level accuracy가 얼마나 개선되었는지를 측정한다.

본 과제에서 중요한 점은 leaderboard 성능만이 아니라 cross-validation 또는 held-out validation에서의 일반화 가능성이다. 특히 MFR과 같은 dictionary 기반 baseline은 training data에 등장한 frequent mapping에는 강하지만, unseen noisy form이나 문맥 의존적 normalization에는 취약할 수 있다. 따라서 본 연구는 단순 baseline 재현에서 시작하여, 언어별 오류 유형과 일반화 한계를 분석하는 것을 첫 번째 목표로 한다.

본 보고서의 기여는 다음과 같이 정리할 수 있다.

- MultiLexNorm2026 제공 baseline인 MFR을 재현하고 언어별 성능을 분석한다. (TBD)
- MFR이 잘 해결하는 normalization 유형과 실패하는 유형을 분류한다. (TBD)
- TBD 모델/규칙/ensemble 접근을 validation 설정에서 비교한다. (TBD)
- 최종 submission 생성 과정과 재현 가능한 실행 명령을 문서화한다. (TBD)

## 2. Related Work

Lexical normalization은 noisy user-generated text를 표준형으로 변환한다는 점에서 grammatical error correction(GEC), spelling correction, text normalization, machine translation과 관련이 있다. 그러나 MultiLexNorm 설정에서는 문장 전체를 새로 쓰기보다 입력 token 또는 어휘 단위의 replacement를 예측한다는 점이 중요하다.

MultiLexNorm 2021 shared task는 여러 언어의 social media dataset을 통합된 평가 방식으로 비교하기 위해 제안되었다. Van der Goot et al. (2021)은 lexical normalization benchmark를 구성하고 intrinsic evaluation으로 ERR을, extrinsic evaluation으로 dependency parsing 및 POS tagging 성능을 사용하였다. 이 shared task는 9개 팀, 18개 submission을 포함했으며, neural normalization system이 기존 rule/dictionary 기반 시스템보다 높은 intrinsic 성능을 보였다.

Samuel and Straka (2021)의 UFAL 시스템은 MultiLexNorm 2021 우승 시스템이다. 이들은 byte-level multilingual sequence-to-sequence model인 ByT5를 사용하고, clean Wikipedia text에 synthetic noise를 주입해 추가 pretraining한 뒤 authentic normalization data로 fine-tuning하였다. 이 접근은 subword vocabulary에 의존하지 않기 때문에 오탈자, 축약, accent omission, 문자 반복 등 noisy surface form에 강하다.

MFR baseline도 중요한 비교 대상이다. MFR은 training data에서 각 input token의 가장 빈번한 normalized output을 사전으로 저장한 뒤 test token에 적용한다. 단순한 방식이지만, lexical normalization에서는 반복적으로 나타나는 비표준-표준 mapping이 많기 때문에 강한 baseline이 될 수 있다. 다만 unseen form, 문맥 의존 ambiguity, split/merge correction, 언어별 형태 변화에는 한계가 있다.

ByT5 (Xue et al., 2022)는 이러한 한계를 보완할 수 있는 pretrained model 후보이다. 대부분의 pretrained language model은 wordpiece나 sentencepiece 같은 subword vocabulary를 사용하지만, ByT5는 UTF-8 byte sequence를 직접 입력과 출력으로 사용한다. 따라서 별도 tokenizer에 없는 noisy form도 byte sequence로 표현할 수 있으며, spelling noise나 문자 단위 변형에 대한 robustness가 높다. MultiLexNorm처럼 정규화 대상이 주로 표면형 변형인 task에서는 이러한 token-free 설계가 특히 적합하다.

다른 관련 접근으로 MoNoise와 CL-MoNoise는 사전, lexical similarity, embedding feature를 활용하여 후보 normalization을 ranking한다. mBART 기반 sentence-level seq2seq 접근은 lexical normalization을 machine translation처럼 문장 변환 문제로 다루지만, MultiLexNorm 2021의 intrinsic word-level evaluation에서는 ByT5 token-level 방식보다 낮은 성능을 보였다. 이 차이는 본 과제에서도 token alignment와 unseen form generation을 동시에 고려해야 함을 시사한다.

## 3. Model

### 3.1 Task Formulation

입력 문장은 noisy token sequence로 주어지고, 시스템은 각 token 또는 token span에 대한 normalized form을 예측한다. 평가에서는 예측 결과와 gold normalization을 비교하여 word-level accuracy를 계산하고, 이를 leave-as-is baseline 대비 ERR로 변환한다.

공식 ERR은 다음과 같이 정의된다.

```text
ERR = (accuracy_system - accuracy_leave-as-is) / (1.0 - accuracy_leave-as-is)
```

본 과제의 최종 ranking은 언어별 ERR의 macro-average를 중심으로 하므로, 데이터 크기가 큰 언어 하나에 과도하게 최적화하기보다 모든 언어에서 안정적인 개선을 얻는 것이 중요하다.

### 3.2 MFR Baseline

MFR은 각 언어의 training split에서 noisy token과 gold normalized token의 mapping 빈도를 계산한다. 동일 noisy token에 여러 normalized form이 등장하면 가장 빈번한 normalized form을 선택한다. Inference 시 dictionary에 있는 token은 해당 normalized form으로 바꾸고, 없는 token은 원형을 유지한다.

장점:

- 구현이 단순하고 재현 가능하다.
- training data에 반복적으로 등장하는 slang, abbreviation, spelling variant에 강하다.
- 모델 학습 비용이 거의 없다.

한계:

- unseen noisy form을 정규화하지 못한다.
- 문맥에 따라 다른 표준형이 필요한 경우를 처리하기 어렵다.
- typo나 character-level 변형을 일반화하지 못한다.
- split/merge처럼 token boundary가 달라지는 현상에 약할 수 있다.

### 3.3 Candidate Neural Approach

TBD.

MFR의 가장 큰 한계는 training data에서 보지 못한 noisy token을 정규화하지 못한다는 점이다. Pretrained seq2seq model은 이 문제를 완화할 수 있다. 특히 ByT5는 byte-level representation을 사용하므로, 표준형과 noisy form 사이의 character/byte-level 변환 패턴을 학습하고 unseen token에도 적용할 가능성이 있다.

참고 접근으로 Samuel and Straka (2021)는 ByT5를 사용해 각 token을 문맥 안에서 독립적으로 정규화하였다. 입력 문장에서 target token의 시작과 끝을 sentinel token으로 표시하고, decoder가 해당 token의 normalized form만 생성하도록 학습했다.

예시:

```text
source sentence: social ppl r gr8
model input: social <X> ppl <Y> r gr8
model output: people
```

이 방식은 token alignment를 유지하기 쉽고 byte-level model의 장점을 활용할 수 있지만, 모든 token마다 별도 input을 만들어야 하므로 inference cost가 크다. 17개 언어를 다루는 MultiLexNorm2026에서는 언어별 model, multilingual single model, routed system 사이의 trade-off를 신중히 비교해야 한다.

본 과제에서 검토할 수 있는 입력 형식 후보는 다음과 같다. 실제 실험 여부는 별도로 결정한다.

- token only: `raw_token -> norm_token`
- language-aware token: `<lang=ko> raw_token -> norm_token`
- context-marked token: `left context <X> raw_token <Y> right context -> norm_token`

Token-only 방식은 빠르지만 문맥을 사용하지 못한다. Context-marked 방식은 UFAL 2021과 가장 유사하며 문맥을 활용할 수 있지만, sentence의 모든 token을 별도 예제로 만들어야 하므로 추론 비용이 크다.

## 4. Experiments

### 4.1 Data

TBD.

주의: Paper Writing Guideline은 참고 논문 맥락에서 MultiLexNorm 2021 설명을 요구하지만, 본 과제의 실제 실험 대상은 MultiLexNorm2026이다. 따라서 Related Work에서는 MultiLexNorm 2021을 설명하고, Experiments에서는 MultiLexNorm2026의 17개 언어 dataset과 public/dev/full split을 명확히 구분한다.

기록할 항목:

- 전체 언어 목록
- train/dev/test 또는 제공 split 구조
- 언어별 sentence/token 수
- normalization 비율
- split/merge 포함 여부
- capitalization correction 포함 여부
- code-switching 여부

### 4.2 Evaluation

공식 평가는 ERR을 사용한다. 본 보고서에서는 leaderboard score와 별도로 validation 또는 cross-validation 성능을 기록한다. 이는 public leaderboard에 과적합하지 않고 generalization을 확인하기 위함이다.

기록할 항목:

- validation split 방식: TBD
- cross-validation 사용 여부: TBD
- official scorer 명령어: TBD
- submission 파일 생성 명령어: TBD

### 4.3 Baselines

본 연구의 첫 번째 실험은 제공 MFR baseline 재현이다. MFR은 성능 개선 목적보다 baseline 재현과 한계 분석을 위한 기준점으로 사용한다.

실험 명령어:

```bash
cd /home/raymond/new_project/external/MultiLexNorm2026

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

결과 기록:

| System | Validation ERR | Public LB ERR | Notes |
| --- | ---: | ---: | --- |
| LAI | TBD | TBD | leave-as-is |
| MFR | 39.02 | TBD | provided baseline, multilingual single dictionary |

### 4.4 Proposed Systems

TBD.

새로운 실험 접근법은 사용자 확인 후 추가한다.

## 5. Results

TBD.

### 5.1 Overall Results

| System | Macro ERR | Notes |
| --- | ---: | --- |
| LAI | 0.00 | by definition |
| MFR | 39.02 | validation, multilingual single dictionary |
| ByT5 language-specific | TBD | completed languages only: de/en positive, ko negative |

### 5.2 Per-language Results

| Language | LAI Acc | MFR ERR | Best System ERR | Main Error Type |
| --- | ---: | ---: | ---: | --- |
| de | 82.04 | 20.85 | 25.43 | ByT5 improves over global MFR by +4.58 ERR |
| en | 93.10 | 28.91 | 44.08 | ByT5 improves over global MFR by +15.17 ERR |
| ko | 91.17 | 11.45 | 11.45 | ByT5 failed to improve over MFR; slang expansion and low changed-token count |

### 5.3 Early ByT5 Language-specific Results

현재까지 validation이 있는 언어 중 `de`, `en`, `ko`에 대해 언어별 ByT5 1 epoch fine-tuning과 global MFR 비교를 완료했다. 이때 MFR은 전체 train split으로 만든 multilingual dictionary를 사용하고, ByT5는 각 언어의 token-level examples만 사용하여 별도의 checkpoint를 학습했다.

독일어와 영어에서는 ByT5가 MFR보다 높은 ERR을 기록했다. 독일어는 LAI 82.04, MFR ERR 20.85, ByT5 ERR 25.43으로 ByT5가 MFR보다 +4.58 ERR 높았다. 영어는 LAI 93.10, MFR ERR 28.91, ByT5 ERR 44.08로 ByT5가 MFR보다 +15.17 ERR 높아 현재까지 가장 뚜렷한 개선을 보였다.

반면 한국어에서는 ByT5가 MFR보다 훨씬 낮은 성능을 보였다. 따라서 동일한 ByT5 fine-tuning 방식이라도 언어별 데이터 크기, changed token 유형, slang/abbreviation expansion 비율에 따라 효과가 크게 달라진다. 현재 결과는 ByT5를 모든 언어에 일괄 적용하기보다 언어별 validation 성능을 기준으로 MFR과 ByT5 중 더 나은 prediction source를 선택하는 routed system의 필요성을 시사한다.

영어 validation sample을 보면 ByT5는 `bruh -> brother`와 같은 일부 informal expression을 정확히 복원했다. 그러나 `yo -> your`를 `you`로 예측하거나 `dese -> these`를 그대로 복사하는 사례도 있어, slang expansion과 spelling correction 모두에서 남은 오류가 있다.

### 5.4 Korean ByT5 Case Study

한국어는 과제에서 강조되는 언어 중 하나이며, validation split에 포함되어 있다. 한국어 train에는 1,701 sentences, 13,130 tokens가 있고, 이 중 실제로 정규화가 필요한 changed token은 958개이다. Validation에는 212 sentences, 1,880 tokens가 있고 changed token은 166개이다.

MFR baseline은 한국어 validation에서 LAI accuracy 91.17, MFR accuracy 92.18, ERR 11.45를 기록했다. 반면, `google/byt5-small`을 UFAL 2021 방식의 context-marked token-level input으로 fine-tuning한 한국어 전용 모델은 MFR을 개선하지 못했다.

한국어 1 epoch ByT5 실험 설정:

```text
model: google/byt5-small
training examples: sample_data/byt5/lang/train_ko_examples.jsonl
input format: left context <extra_id_0> raw_token <extra_id_1> right context
target format: norm_token
max steps: 3300
batch size: 4
learning rate: 5e-5
```

학습 결과:

```text
first_loss: 3.240163
last_loss: 0.532505
avg_last_10: 0.056538
```

Validation 결과:

```text
LAI accuracy: 91.17
Model accuracy: 80.48
ERR: -121.08
```

오류 분석 결과, changed token 166개 중 정확히 맞춘 token은 0개였다. 대부분의 changed token에서는 raw token을 그대로 복사했고, unchanged token 1,714개 중 201개는 불필요하게 다른 token으로 바꾸었다.

```text
changed accuracy: 0 / 166 = 0.00%
unchanged accuracy: 1513 / 1714 = 87.11%
```

이 결과는 한국어 subset에서 direct ByT5 fine-tuning이 충분하지 않음을 보여준다. 한국어 normalization에는 `여친 -> 여자친구`, `ㄹㅇ -> 진짜`, `띵작 -> 명작`처럼 단순 철자 보정보다 slang/abbreviation expansion이 많고, train changed token 수가 958개로 적다. 따라서 ByT5가 byte-level spelling pattern만으로 일반화하기 어렵고, MFR이 low-resource setting에서 더 안정적인 baseline으로 작동했다.

추가로 MFR을 기본값으로 두고 ByT5가 confidence margin을 넘을 때만 prediction을 덮어쓰는 hybrid system을 평가했다. 그러나 best threshold에서도 한국어 hybrid는 MFR과 동일한 accuracy 92.18, ERR 11.45에 머물렀다. 이는 현재 한국어 ByT5 checkpoint가 MFR보다 나은 correction을 제공하지 못했음을 의미한다.

한국어에서 이 방식이 특히 실패한 이유는 다음과 같이 분석된다. 첫째, 한국어 train changed token은 958개로 영어 2,666개, 독일어 2,578개보다 훨씬 적다. 둘째, validation changed token의 raw form이 global MFR dictionary에 등장한 비율은 한국어 36.1%로, 영어 80.1%, 독일어 67.9%보다 낮다. 같은 언어 train changed pair에 exact match로 등장한 비율도 한국어 26.5%, 영어 73.5%, 독일어 49.7%로 차이가 크다. 즉 한국어 validation은 현재 train split에서 반복적으로 관찰된 normalization pair가 적어, 1 epoch fine-tuning만으로 일반화하기 어렵다.

셋째, 한국어 normalization은 spelling correction만이 아니라 slang/profanity/abbreviation의 의미적 치환을 많이 포함한다. 예를 들어 `존나 -> 매우`, `ㄹㅇ -> 진짜`, `시발 -> 이런`, `여친이랑 -> 여자친구랑`, `띵작이다 -> 명작이다`, `개씹존잘 -> 매우 잘생긴 사람`과 같은 사례에서는 표면형 유사성만으로 정답을 생성하기 어렵다. 특히 한국어는 조사와 어미가 같은 eojeol token에 붙어 있으므로, 모델은 내부 어휘를 바꾸면서 suffix를 보존하거나 함께 변형해야 한다.

넷째, byte-level model인 ByT5에서는 한국어 입력이 영어/독일어보다 길어진다. 현재 설정의 `max_input_length=256`에서 한국어 train examples의 13.2%, changed examples의 5.2%가 길이 제한을 넘었지만, 영어와 독일어는 해당 비율이 0.0%였다. 따라서 일부 한국어 example에서는 target token 주변 문맥이 더 자주 잘렸을 가능성이 있다. Target length truncation은 changed examples 기준 0.9%로 작았지만, input truncation은 한국어 성능 저하의 한 원인일 수 있다.

### 5.5 Error Analysis

분석할 오류 유형:

- frequent abbreviation
- spelling typo
- repeated character
- capitalization
- accent/diacritic omission
- split/merge
- unseen noisy form
- ambiguous token requiring context
- code-switching

MFR이 해결하는 경우:

- TBD.

MFR이 실패하는 경우:

- TBD.

Neural/rule-based system이 추가로 해결하는 경우:

- TBD.

## 6. Discussion

TBD.

현재까지 문헌 조사에서 얻은 시사점은 다음과 같다. MultiLexNorm 2021에서 MFR은 단순하지만 강한 baseline이었다. 이는 많은 lexical normalization 사례가 반복적인 token replacement로 해결된다는 뜻이다. 그러나 UFAL의 ByT5 기반 시스템은 byte-level representation과 synthetic pretraining을 통해 MFR보다 훨씬 높은 ERR을 달성했다. 따라서 MultiLexNorm2026에서도 MFR의 한계를 언어별로 확인한 뒤, character/byte-level 일반화가 필요한 부분을 찾는 것이 중요하다.

특히 macro-average ERR은 모든 언어를 동일하게 반영하므로, resource가 많거나 쉬운 언어의 점수만 높이는 전략은 충분하지 않을 수 있다. 언어별 normalization 비율과 오류 유형을 분석하여 약한 언어를 targeted하게 개선하는 방식이 필요하다.

## 7. Conclusion

TBD.

본 보고서는 MultiLexNorm2026 과제에서 MFR baseline 재현을 출발점으로 multilingual lexical normalization 시스템을 분석한다. 향후 작업은 데이터 로딩 확인, MFR 재현, 언어별 오류 분석, validation 기반 일반화 평가, 최종 submission 생성 순서로 진행한다.

## References 후보

- Rob van der Goot et al. 2021. MultiLexNorm: A Shared Task on Multilingual Lexical Normalization.
- David Samuel and Milan Straka. 2021. UFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5.
- Linting Xue et al. 2021. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.
- Rob van der Goot. 2019. MoNoise: A Multi-lingual and Easy-to-use Lexical Normalization Tool.
- Rob van der Goot. 2021. CL-MoNoise: Cross-lingual Lexical Normalization.
- Ana-Maria Bucur and Liviu P. Dinu. 2021. Sequence-to-Sequence Lexical Normalization with Multilingual Transformers.

## TODO

- Team name 확정.
- leaderboard 계정 이메일 기록. CodaBench 등록은 university email(`g.skku.edu`) 사용.
- 데이터 split과 파일 format 확인.
- MFR baseline 실행 명령어 기록.
- validation/cross-validation 설계 확정.
- 언어별 결과 표 채우기.
- 오류 분석 예시 수집.
- 영어 ACL LaTeX 원고로 이전.
- `requirements.txt` 포함 여부 확인.
- 최종 `TeamName_CodeFile.zip` 구성 확인.
- 개인별 team work report 작성 항목 정리.
