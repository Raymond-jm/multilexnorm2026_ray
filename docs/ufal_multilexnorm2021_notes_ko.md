# UFAL at MultiLexNorm 2021 논문 정리

대상 논문: David Samuel and Milan Straka. 2021. "UFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5." W-NUT 2021.

관련 overview: Rob van der Goot et al. 2021. "MultiLexNorm: A Shared Task on Multilingual Lexical Normalization." W-NUT 2021.

## 1. 이 대회가 다루는 문제

MultiLexNorm은 SNS, Twitter, 온라인 커뮤니티처럼 비표준 표현이 많이 등장하는 텍스트를 표준형으로 바꾸는 lexical normalization shared task이다. 예를 들어 `ppl`을 `people`, `r`을 `are`, `gr8`을 `great`처럼 바꾸는 문제다.

핵심은 문장 전체를 자연스럽게 다시 쓰는 것이 아니라, 입력 토큰 단위의 비표준 어휘를 canonical form으로 정규화하는 것이다. 따라서 맞춤법 교정, 문법 오류 수정, 기계번역과 비슷한 면이 있지만 평가와 데이터 구조는 token-to-token lexical replacement에 더 가깝다.

2021 대회는 11개 언어의 12개 social media dataset을 사용했다. 여기에는 Danish, German, English, Spanish, Croatian, Indonesian-English code-switching, Italian, Dutch, Serbian, Slovenian, Turkish, Turkish-German code-switching이 포함된다. 일부 데이터셋은 split/merge token correction과 capitalization correction을 포함한다.

2026 과제는 AGENTS.md 기준으로 17개 언어를 포함하므로, 2021 논문의 수치와 언어 구성은 그대로 복제 대상이 아니라 방법론 참고 자료로 봐야 한다.

## 2. 평가 방식

주요 intrinsic metric은 ERR(Error Reduction Rate)이다. leave-as-is, 즉 아무 것도 정규화하지 않는 baseline 대비 word-level accuracy가 얼마나 개선되었는지를 측정한다.

논문에 제시된 공식은 다음과 같다.

```text
ERR = (accuracy_system - accuracy_leave-as-is) / (1.0 - accuracy_leave-as-is)
```

최종 ranking은 dataset별 ERR을 macro-average해서 결정된다. 이 점은 언어별 데이터 크기가 큰 언어에만 잘 맞추는 방식보다, 각 언어에서 고르게 개선되는 시스템이 중요하다는 뜻이다.

2021 대회에는 extrinsic evaluation도 있었다. 정규화된 데이터를 dependency parsing에 넣고 LAS(Label Attachment Score)를 측정했다. Overview 논문 기준으로 POS tagging도 다뤄졌지만, UFAL 논문은 dependency parsing 중심으로 설명한다.

## 3. baseline과 대회 난점

논문에서 비교한 baseline은 다음과 같다.

- LAI: Leave-as-is, 아무 단어도 바꾸지 않음
- MFR: training data에서 input token별 가장 자주 등장한 replacement를 사용하는 dictionary baseline
- MoNoise: 사전, FastText embedding, hand-crafted feature를 활용하는 기존 multilingual lexical normalization tool

MFR은 단순하지만 강하다. 특히 2021 intrinsic evaluation에서 평균 ERR 38.37을 기록했고, English와 Indonesian-English에서는 각각 64.9, 61.2로 꽤 높았다. 이는 lexical normalization이 상당 부분 frequent replacement/dictionary lookup으로 해결된다는 뜻이다.

하지만 MFR의 한계도 분명하다.

- unseen noisy form에는 약하다.
- 문맥에 따라 다른 정규화가 필요한 경우를 처리하기 어렵다.
- capitalization, split/merge, typo, 축약, 언어별 특수 변형을 일반화하기 어렵다.
- public leaderboard에만 맞춘 replacement dictionary는 cross-validation/generalization에서 흔들릴 수 있다.

## 4. UFAL의 핵심 아이디어

UFAL은 ByT5-small을 기반으로 한 byte-level seq2seq 접근을 사용했다. ByT5는 subword tokenizer가 아니라 UTF-8 byte sequence를 직접 처리하기 때문에 noisy text, typo, 비표준 철자, 다국어 문자 변형에 상대적으로 유리하다.

모델 입력은 문장 전체를 한번에 normalized sentence로 바꾸는 방식이 아니다. 각 input word마다 별도 예제를 만들고, 해당 word의 시작과 끝을 sentinel token으로 표시한다. 모델은 marked word 하나의 normalized output만 생성한다.

예시 개념:

```text
원문: social ppl r gr8
입력: social <X> ppl <Y> r gr8
출력: people
```

이 방식의 장점은 token-level alignment를 유지하기 쉽고, ByT5의 span reconstruction pretraining과 어느 정도 유사한 형태를 만든다는 점이다. 반면 문장 내 모든 단어마다 별도 input을 만들어야 하므로 inference가 비효율적이다.

## 5. 학습 절차

UFAL의 학습은 세 단계다.

1. 기본 ByT5 self-supervised pretraining: 원래 Google ByT5가 mC4 등에서 학습한 상태를 사용한다.
2. synthetic lexical normalization pretraining: Wikipedia clean text에 인위적 noise를 넣어 noisy-to-clean 데이터를 만든 뒤 추가 pretraining한다.
3. authentic training data fine-tuning: MultiLexNorm의 hand-annotated data로 fine-tuning한다.

Synthetic noise는 training data에서 관찰된 패턴을 바탕으로 만든다.

- training data에 등장한 normalized output과 noisy input mapping을 이용한 replacement
- accent mark removal
- capitalization change
- apostrophe removal
- keyboard typo, character deletion/insertion/substitution/swap
- 언어별 특수 규칙, 예: Indonesian plural 표기 `laki-lakinya` -> `laki2nya`
- split/merge word
- vowel omission, prefix shortening, repeated character

중요한 점은 synthetic pretraining이 완전한 unsupervised 방식이 아니라는 것이다. noise 확률과 규칙을 training data에서 추정했기 때문에, 새 언어에 일반화하려면 해당 언어의 annotated lexical normalization data나 전문가 지식이 필요하다.

## 6. 실험 설정

2021 UFAL 논문의 주요 설정은 다음과 같다.

- 모델: ByT5-small, 300M parameters
- 언어/데이터셋별 independent model 학습
- synthetic pretraining: 최소 100,000 optimizer steps
- optimizer: AdaFactor
- batch size: 128
- pretraining learning rate: inverse square root decay, peak LR 5e-4, warmup 4,000 steps
- fine-tuning: constant LR 1e-4, 50 epochs
- competition model은 train+dev를 합치고, 3%만 development check용으로 분리
- mixed fine-tuning은 authentic data와 synthetic data를 1:1로 섞는 방식이며, dev 성능이 좋아질 때만 사용
- 제출은 single model과 4-model ensemble 두 가지

Ensemble은 각 word에 대해 beam search로 후보 16개를 만들고, 4개 모델의 평균 probability가 가장 높은 replacement를 선택한다.

## 7. 주요 결과

Intrinsic evaluation에서 UFAL은 압도적으로 우수했다.

- UFAL single: macro ERR 66.21
- UFAL ensemble: macro ERR 67.30
- 2위 HEL-LJU: macro ERR 53.58
- MFR baseline: macro ERR 38.37
- LAI baseline: macro ERR 0.00

Ablation에서 중요한 포인트는 다음과 같다.

- mT5-small only fine-tuning: ERR 33.62
- ByT5-small only fine-tuning: ERR 59.23
- ByT5 + synthetic pretraining + train fine-tuning: ERR 64.88
- ByT5 + synthetic pretraining + train+dev fine-tuning: ERR 66.21
- 4-model ensemble: ERR 67.30

즉, 성능 기여는 대략 `byte-level model 선택`이 가장 크고, 그 다음이 synthetic pretraining, train+dev 활용, ensemble 순서로 보인다. beam search 자체는 greedy decoding과 거의 차이가 없었다.

Extrinsic evaluation에서는 UFAL이 평균 LAS 64.17로 1위였지만, 차이는 intrinsic ERR보다 작았다. 흥미롭게도 MFR도 extrinsic 평균 LAS 63.31로 강하게 나왔다. 이는 downstream parsing에서는 완벽한 lexical normalization이 아니어도 자주 등장하는 표준화만으로 상당한 이득을 줄 수 있음을 시사한다.

## 8. 한계점

UFAL 논문에서 직접 드러나는 한계와 우리가 주의해야 할 점은 다음과 같다.

- 단어마다 문장을 다시 encoding해야 하므로 inference가 비효율적이다.
- 언어별 independent model을 학습했기 때문에 17개 언어 과제에서는 학습/추론 비용이 커질 수 있다.
- synthetic noise 생성 규칙과 확률이 training data에 의존한다.
- mixed fine-tuning을 dev 성능으로 선택하는 전략은 실제 ablation에서 base fine-tuning보다 나빴다.
- ensemble gain은 약 1 ERR point로 존재하지만, 4배 가까운 추론 비용을 감수해야 한다.
- pretraining-only 성능은 낮다. synthetic data만으로는 부족하고 authentic fine-tuning이 필수다.
- sentence-level generation을 피했기 때문에 token 간 의존성이 강한 normalization에는 약할 수 있다.
- split/merge는 다루지만, token-level independent prediction 구조에서는 문장 전체 coherence를 직접 최적화하지 않는다.

논문은 future work로 input sentence를 한 번만 encoding하는 구조를 제안한다. 예를 들어 문장 전체를 한 번 decode하거나, 모든 input word를 서로 다른 sentinel token으로 분리한 뒤 sentinel별 decoder initialization을 사용하는 방식이다.

## 9. 우리 과제에 주는 시사점

현재 과제에서 바로 가져올 수 있는 방향은 다음과 같다.

- MFR은 반드시 재현해야 한다. 단순 baseline이지만 강하고, 언어별 한계 분석에 적합하다.
- ByT5 계열은 noisy multilingual lexical normalization에서 강한 baseline 후보이다.
- 2026 과제는 17개 언어이므로, 언어별 model과 multilingual single model을 비교할 가치가 있다. 다만 새 실험 접근으로 계획에 넣기 전 사용자 확인이 필요하다.
- ERR macro-average가 핵심이므로 low-resource/어려운 언어의 개선이 전체 점수에 중요하다.
- public leaderboard 과적합을 피하려면 validation 또는 cross-validation 기록이 반드시 필요하다.
- synthetic data는 강력하지만, target language lexical-normalization 추가 데이터 사용 제한을 확인하고 안전하게 설계해야 한다.
- 보고서에는 단순 점수보다 왜 MFR이 잘 되는 언어와 안 되는 언어가 갈리는지 분석하는 것이 중요하다.

## 10. 보고서 초안에 넣을 수 있는 문장

TBD: 우리 실험 결과를 넣은 뒤 수정한다.

Lexical normalization은 noisy social media text에 나타나는 비표준 어휘를 표준형으로 변환하는 작업이다. MultiLexNorm 2021은 여러 언어와 code-switching 환경을 포함한 benchmark를 제시하고, ERR을 사용하여 leave-as-is baseline 대비 오류 감소율을 측정하였다. Samuel and Straka (2021)는 byte-level sequence-to-sequence model인 ByT5를 사용하여 이 문제를 token-level generation 문제로 정식화하였다. 이 접근은 subword vocabulary에 의존하지 않기 때문에 오탈자, 축약, 문자 반복, accent omission 등 noisy text의 표면 변형에 강하다는 장점이 있다.

UFAL 시스템의 핵심은 synthetic noise pretraining과 authentic normalization data fine-tuning의 결합이다. Clean Wikipedia text에 training data에서 관찰된 replacement, typo, capitalization, split/merge 등의 noise를 주입하여 noisy-to-clean 학습 예제를 생성하고, 이후 실제 annotated data로 fine-tuning한다. Ablation 결과는 ByT5가 mT5보다 lexical normalization에 훨씬 적합하며, synthetic pretraining이 추가적인 성능 향상을 제공함을 보인다.

그러나 이 시스템은 각 token마다 별도의 model input을 생성하기 때문에 inference cost가 높고, synthetic noise 설계가 training data의 통계와 언어별 규칙에 의존한다. 또한 언어별 independent model을 학습하는 방식은 언어 수가 늘어나는 환경에서 유지 비용이 커질 수 있다. 따라서 본 과제에서는 MFR baseline을 먼저 재현하고 언어별 오류 유형을 분석한 뒤, byte-level seq2seq 모델과 multilingual/routed system의 trade-off를 실험적으로 검토할 필요가 있다.

## 11. 참고 링크

- UFAL paper: https://aclanthology.org/2021.wnut-1.54/
- UFAL PDF: https://aclanthology.org/2021.wnut-1.54.pdf
- MultiLexNorm overview: https://aclanthology.org/2021.wnut-1.55/
- 2021 task page: https://noisy-text.github.io/2021/multi-lexnorm.html
- UFAL code: https://github.com/ufal/multilexnorm2021
- UFAL Hugging Face models: https://huggingface.co/ufal
