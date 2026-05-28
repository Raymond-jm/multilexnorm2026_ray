# Pretrained Model 관련 논문 정리

목적: MFR baseline의 unseen noisy form 한계를 pretrained sequence-to-sequence model, 특히 ByT5 계열로 어떻게 보완할 수 있는지 정리한다.

읽은/확인한 자료:

- Samuel and Straka (2021), "UFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5"
- van der Goot et al. (2021), "MultiLexNorm: A Shared Task on Multilingual Lexical Normalization"
- Xue et al. (2022), "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
- van der Goot (2019), "MoNoise: A Multi-lingual and Easy-to-use Lexical Normalization Tool"
- van der Goot (2021), "CL-MoNoise: Cross-lingual Lexical Normalization"
- Bucur and Dinu (2021), "Sequence-to-Sequence Lexical Normalization with Multilingual Transformers"
- Lourentzou et al. (2019), "Adapting Sequence to Sequence Models for Text Normalization in Social Media"

## 1. 왜 MFR만으로 부족한가

MFR(Most-Frequent-Replacement)은 training data에 등장한 `raw -> norm` mapping을 기억하고, 같은 raw token이 다시 나오면 가장 자주 나온 norm token으로 치환한다. 이 방식은 반복적으로 등장하는 slang, 약어, 철자 변형에는 강하다.

하지만 MFR은 본질적으로 memorization 방식이다.

- train에 없는 noisy form은 정규화하지 못한다.
- `bcuz`, `bcoz`, `becuz`, `bcs`처럼 같은 standard word를 향하는 변형을 하나의 패턴으로 일반화하지 못한다.
- accent omission, repeated character, keyboard typo, vowel omission처럼 character-level 규칙성이 있는 변형을 학습하지 못한다.
- 같은 surface token이 문맥에 따라 다른 normalization을 가질 때 문맥을 쓰지 못한다.
- 한국어, 일본어, 태국어, 베트남어처럼 tokenization과 문자 단위 변형이 복잡한 언어에서는 word-level dictionary만으로 부족할 가능성이 크다.

따라서 MFR은 강한 baseline이지만, unseen data/generalization 평가에서 약점이 드러날 수 있다. 과제 guideline이 cross-validation/generalization을 강조하는 것도 이 점과 연결된다.

## 2. ByT5가 이 문제에 맞는 이유

ByT5는 subword token을 사용하지 않고 UTF-8 byte sequence를 직접 입력/출력으로 처리하는 T5 계열 pretrained seq2seq model이다. ByT5 논문은 token-free model의 장점으로 다음을 강조한다.

- 별도 tokenizer나 vocabulary 없이 여러 언어의 raw text를 처리할 수 있다.
- 오탈자와 비표준 표면형에 더 강하다.
- spelling, pronunciation처럼 문자/소리 단위 정보가 중요한 task에서 유리하다.
- preprocessing/tokenization pipeline의 오류 가능성을 줄인다.

Lexical normalization은 바로 이런 특성을 요구한다. noisy token은 대개 표준어와 완전히 무관한 새 단어라기보다, 표준형의 character/byte-level 변형인 경우가 많다.

예시:

```text
tilfaeldigt -> tilfældigt
skaelver    -> skælver
vaek        -> væk
bcause      -> because
u           -> you
ㅋㅋㅋㅋ      -> ㅋㅋ 또는 웃음 표현 정규화 후보
```

MFR은 `skaelver -> skælver`를 training에서 본 적이 있을 때만 바꿀 수 있지만, ByT5는 `ae -> æ` 같은 byte/character pattern을 여러 예시에서 학습하면 unseen word에도 적용할 수 있다.

## 3. UFAL 2021 시스템의 핵심 설계

UFAL은 MultiLexNorm 2021 우승 시스템으로, ByT5-small을 lexical normalization에 맞게 fine-tuning했다. 핵심은 문장 전체를 한 번에 normalized sentence로 생성하지 않고, target token 하나를 문맥 속에서 표시한 뒤 해당 token의 normalized form만 생성하는 방식이다.

개념 예시:

```text
source sentence: social ppl r gr8
model input:     social <X> ppl <Y> r gr8
model output:    people
```

이 설계의 장점:

- token-level evaluation과 alignment가 쉽다.
- target token 주변 문맥을 사용할 수 있다.
- ByT5의 byte-level representation으로 typo/철자 변형을 처리할 수 있다.
- output이 한 token의 normalized form이라 sentence-level generation보다 통제하기 쉽다.

학습은 세 단계다.

1. pretrained ByT5-small 사용
2. Wikipedia clean text에 synthetic noise를 넣어 추가 pretraining
3. authentic MultiLexNorm training data로 fine-tuning

Synthetic noise는 accent removal, capitalization change, apostrophe removal, keyboard typo, deletion/insertion/substitution/swap, repeated character, vowel omission, split/merge 등 noisy social text에 맞춘 변형을 포함한다.

## 4. 2021 결과에서 배울 점

MultiLexNorm 2021에서 MFR baseline은 이미 꽤 강했다. 이는 lexical normalization의 많은 부분이 frequent replacement로 해결된다는 뜻이다. 그러나 UFAL ByT5 시스템은 MFR보다 훨씬 높은 intrinsic ERR을 기록했다.

논문 ablation의 중요한 메시지:

- mT5-small보다 ByT5-small이 훨씬 강했다.
- ByT5 only fine-tuning도 강했지만, synthetic pretraining이 추가 성능을 제공했다.
- ensemble은 성능을 조금 더 올렸지만 비용이 크다.
- beam search는 greedy decoding 대비 큰 이득이 없었다.

해석:

- 이 task에서는 word/subword vocabulary보다 byte-level modeling이 핵심일 수 있다.
- pretrained model이 이미 multilingual/character pattern 지식을 갖고 있고, fine-tuning이 이를 normalization task에 맞춘다.
- synthetic noise는 unseen noisy form 일반화에 도움이 된다.

## 5. 다른 관련 접근

### MoNoise / CL-MoNoise

MoNoise는 사전, lexical similarity, word embedding, feature 기반 ranking을 조합하는 multilingual lexical normalization tool이다. CL-MoNoise는 이를 cross-lingual setting에 맞춰 확장했다.

장점:

- lightweight하고 해석 가능하다.
- 사전/embedding 후보 생성이 강한 언어에서는 효과적이다.

한계:

- 후보 생성이 실패하면 정답을 찾기 어렵다.
- 언어별 resource와 feature 설계에 의존한다.
- pretrained seq2seq model처럼 새로운 string을 자유롭게 생성하는 능력은 제한적이다.

### mBART sentence-level seq2seq

Bucur and Dinu (2021)는 multilingual pretrained transformer인 mBART를 사용해 문장 전체 lexical normalization을 machine translation처럼 처리했다. 기술적으로 단순하고 pretrained multilingual model을 활용할 수 있지만, word-level intrinsic score는 최고 시스템보다 낮았다. 대신 downstream task에서는 raw text보다 개선되는 효과가 있었다.

이 접근은 우리 과제에서 비교 후보가 될 수 있지만, token-level prediction과 submission format을 맞추기 위한 후처리가 필요할 수 있다.

### Hybrid word-character seq2seq

Lourentzou et al. (2019)는 social media text normalization에서 문맥 정보가 중요하다고 보고, word-character attention encoder-decoder와 synthetic adversarial examples를 사용했다. 이는 UFAL의 synthetic noise pretraining과 방향이 비슷하다. 차이는 최근에는 ByT5처럼 이미 byte-level로 pretraining된 model을 가져와 fine-tuning할 수 있다는 점이다.

## 6. 우리 과제 전략으로 연결

현재 데이터 확인 결과, MultiLexNorm2026 공개 데이터는 모든 split에서 `len(raw) == len(norm)`이다. 따라서 token-level prediction 방식이 자연스럽다.

권장 실험 흐름은 다음과 같다. 단, 실제 실행 전 사용자 확인이 필요하다.

1. MFR baseline으로 validation ERR과 언어별 한계 확인
2. MFR이 실패하는 token을 seen/unseen으로 분류
3. unseen changed token 비율과 언어별 분포 분석
4. ByT5-small 또는 google/byt5-small 기반 token-level fine-tuning 설계
5. 입력 format 비교:
   - token only: `raw_token`
   - language + token: `<lang=ko> raw_token`
   - context-marked token: `left context <X> token <Y> right context`
6. 먼저 작은 언어 subset 또는 validation이 있는 언어에서 smoke test
7. 성능이 의미 있으면 전체 multilingual fine-tuning으로 확장

가장 보고서에 설득력 있는 논리는 다음과 같다.

> MFR provides a strong memorization baseline, but it cannot normalize unseen noisy forms. Byte-level pretrained seq2seq models such as ByT5 address this limitation by modeling lexical variation below the word/subword level and by generating normalized forms compositionally. Following the UFAL MultiLexNorm 2021 system, we therefore treat lexical normalization as token-level generation with sentence context.

## 7. 실험 설계상 주의점

- ByT5-small도 300M parameter라 compute 비용이 있다.
- 모든 token을 예제로 만들면 데이터가 sentence 수보다 훨씬 커진다.
- unchanged token이 많으므로 changed/unchanged sampling strategy가 필요할 수 있다.
- context-marked 방식은 inference 때 sentence의 모든 token에 대해 forward pass가 필요해 느리다.
- token-only 방식은 빠르지만 문맥 의존 normalization을 못 본다.
- language-specific model은 성능이 좋을 수 있지만 17개 언어에서 비용이 크다.
- multilingual single model은 효율적이지만 언어별 특수 패턴이 희석될 수 있다.
- public leaderboard보다 validation/cross-validation 성능과 오류 분석을 우선 기록해야 한다.

## 8. 참고 링크

- UFAL at MultiLexNorm 2021: https://aclanthology.org/2021.wnut-1.54/
- MultiLexNorm 2021 overview: https://aclanthology.org/2021.wnut-1.55/
- ByT5: https://aclanthology.org/2022.tacl-1.17/
- MoNoise: https://aclanthology.org/P19-3032/
- CL-MoNoise: https://aclanthology.org/2021.wnut-1.56/
- mBART seq2seq lexical normalization: https://arxiv.org/abs/2110.02869
- Social media seq2seq normalization: https://ojs.aaai.org/index.php/ICWSM/article/view/3234
