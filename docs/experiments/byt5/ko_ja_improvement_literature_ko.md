# Korean/Japanese 성능 개선을 위한 문헌 및 데이터 분석

작성일: 2026-05-29

목표: MultiLexNorm2026에서 한국어(`ko`)와 일본어(`ja`) 성능을 개선하기 위해, validation 정답을 외우지 않는 방식의 후보 방법을 정리한다.

## 원칙

- Validation 정답 pair를 직접 rule/dictionary에 추가하지 않는다.
- Validation은 오류 유형 분석, 모델 선택, threshold 선택에만 사용한다.
- 실제 개선 규칙은 train split에서 유도하거나, task-specific gold annotation이 아닌 일반 언어 처리 원리/공개 pretrained model을 사용한다.

## 관련 문헌

### 한국어 noisy text

- Lee and Shin (2021), "The Korean Morphologically Tight-Fitting Tokenizer for Noisy User-Generated Texts"
  - URL: https://aclanthology.org/2021.wnut-1.45/
  - 핵심: 한국어 user-generated text는 proper noun, coinage, internet slang을 많이 포함하며, 기존 형태소 분석기나 formal text 기반 LM이 잘 처리하지 못한다.
  - 시사점: 한국어는 subword/byte generation만으로 해결하기보다 eojeol 내부의 slang/coinage와 조사/어미 결합을 고려해야 한다.

- Cho and Kim (2021), "Google-trickers, Yaminjeongeum, and Leetspeak: An Empirical Taxonomy for Intentionally Noisy User-Generated Text"
  - URL: https://aclanthology.org/2021.wnut-1.7/
  - 핵심: online text에서는 의도적 noise가 의미 전달, meme, 특정 집단 내 소통, 우회 표현 등을 위해 사용된다.
  - 시사점: 한국어 validation에서 보이는 욕설/비하/은어 치환은 단순 오탈자 보정보다 의미적 normalization에 가깝다.

### 일본어 noisy text

- Saito et al. (2014), "Morphological Analysis for Japanese Noisy Text based on Character-level and Word-level Normalization"
  - URL: https://aclanthology.org/C14-1167/
  - 핵심: 일본어 noisy text 처리는 lexical normalization과 word segmentation/POS analysis가 강하게 연결된다.
  - 시사점: 일본어는 `てる -> て いる`, `じゃ -> で は`처럼 normalization 결과가 token 내부 rewrite뿐 아니라 segmentation-like output을 포함한다.

- Saito et al. (2017), "Automatically Extracting Variant-Normalization Pairs for Japanese Text Normalization"
  - URL: https://aclanthology.org/I17-1094/
  - 핵심: unsegmented social media text에서 variant-normalization pair를 자동 추출하고 일본어 형태소 분석에 넣으면 normalization recall이 개선된다.
  - 시사점: MultiLexNorm2026 일본어도 train pair에서 문자 유사도 기반 variant-normalization pair를 확장하는 방식이 유망하다.

- Kondo et al. (2025), "Text Normalization for Sentiment Analysis in Japanese Social Media"
  - URL: https://aclanthology.org/2025.wnut-1.16/
  - 핵심: Japanese SNS normalization taxonomy를 33개 editing operation으로 구성했고, normalization이 downstream sentiment analysis에 도움이 됨을 보였다.
  - 시사점: 일본어는 punctuation, spoken-style contraction, kana/kanji/katakana variation 등 operation별 rule 분석이 적합하다.

### 일반 lexical normalization

- Samuel and Straka (2021), "UFAL at MultiLexNorm 2021"
  - URL: https://aclanthology.org/2021.wnut-1.54/
  - 핵심: ByT5를 synthetic data로 추가 pretraining하고 authentic normalization data로 fine-tuning하여 MultiLexNorm 2021 우승.
  - 시사점: 현재 실험은 authentic fine-tuning만 수행했으므로, 한국어/일본어에서 synthetic pretraining 또는 augmentation을 추가하면 개선 가능성이 있다.

- van der Goot (2021), "CL-MoNoise: Cross-lingual Lexical Normalization"
  - URL: https://aclanthology.org/2021.wnut-1.56/
  - 핵심: MoNoise 기반으로 cross-lingual lexical normalization을 시도하며, spelling errors, non-standard words, shortening, capitalization, punctuation 등을 다룬다.
  - 시사점: MFR 단일 dictionary보다 후보 생성 + ranking 구조가 한국어/일본어 rule 확장에 더 적합할 수 있다.

## 데이터 분석 요약

| Metric | ko | ja |
| --- | ---: | ---: |
| Train changed tokens | 958 | 4,565 |
| Validation changed tokens | 166 | 683 |
| Val changed raw seen in global MFR dictionary | 36.1% | 88.1% |
| Val changed exact pair seen in same-language train changed pairs | 26.5% | 73.8% |
| Global MFR correct on changed validation tokens | 16.3% | 32.9% |

## 한국어 해석

한국어는 changed token 수가 적고 validation changed raw가 train/MFR dictionary에 반복 등장하는 비율도 낮다. 또한 `존나 -> 매우`, `ㄹㅇ -> 진짜`, `시발 -> 이런`처럼 slang/profanity/abbreviation을 의미적으로 순화하는 pair가 많다.

따라서 한국어는 ByT5 단독 generation보다 다음 접근이 더 유망하다.

1. Train split 기반 MFR dictionary 유지
2. Train pair에서 base slang replacement를 추출
3. 조사/어미가 붙은 eojeol에서 suffix-preserving replacement 시도
4. 바꿀지 확신이 낮으면 raw/MFR prediction 유지
5. Validation 정답 pair는 직접 추가하지 않음

## 일본어 해석

일본어는 한국어보다 train changed token이 많고 validation changed pair 반복성이 높다. 대표 pair도 `ん -> の`, `てる -> て いる`, `て -> て い`, `… -> … 。`, `じゃ -> で は`처럼 train에서 반복적으로 나타난다.

따라서 일본어는 다음 접근이 유망하다.

1. MFR baseline이 상당히 강할 가능성이 큼
2. ByT5 1epoch를 먼저 평가해 MFR 대비 개선 여부 확인
3. 반복적인 Japanese operation에 대해서는 train-derived rule postprocessing 고려
4. punctuation/sentence-final normalization, spoken contraction expansion, kana/kanji variation을 분리해 오류 분석

## 다음 실험 후보

### 후보 A: 한국어 MFR+suffix rule

- 목적: train에서 배운 base replacement를 조사/어미가 붙은 eojeol에도 적용한다.
- 장점: validation overfitting 위험이 낮고 한국어 실패 원인과 직접 연결된다.
- 위험: 잘못 suffix를 분리하면 over-editing 증가.

### 후보 B: 일본어 ByT5 1epoch 평가

- 목적: 일본어는 반복성이 높으므로 현재 ByT5 방식이 실제로 MFR보다 좋은지 확인한다.
- 장점: 이미 자동화 스크립트가 준비되어 있다.
- 위험: 학습 시간이 길다(`15,476` steps).

### 후보 C: 일본어 train-derived rule postprocessing

- 목적: train에서 반복되는 `てる -> て いる`, `ん -> の`, punctuation expansion 등을 MFR/rule로 안정화한다.
- 장점: 일본어 반복 pair가 많아 효과 가능성이 높다.
- 위험: context-dependent pair(`って -> と/は/という`)는 단순 MFR rule로 틀릴 수 있음.
