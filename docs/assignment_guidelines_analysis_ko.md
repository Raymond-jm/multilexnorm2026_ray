# Assignment Guidelines 분석

읽은 파일:

- `Assignment Guideline.pdf`
- `Assignment Paper Writing Guideline.pdf`

## 1. 과제 핵심

이 과제는 MultiLexNorm2 / MultiLexNorm2026 multilingual lexical normalization task를 기반으로 한다. 목표는 SNS나 온라인 커뮤니티 텍스트에 등장하는 비표준 단어와 표현을 표준형으로 정규화하는 것이다.

공식 task page 기준으로 17개 언어를 포함하며, Thai, Vietnamese, Korean, Japanese, Indonesian 등이 특히 강조된다. 공식 평가는 word-level normalization 성능을 ERR(Error Reduction Rate)로 측정한다.

제공 baseline code에는 다음이 포함된다.

- data loading 예시
- simple MFR(Most-Frequent-Replacement) baseline 실행 예시
- evaluation 예시

학생은 baseline code를 기반으로 작업해야 하며, official task guideline 범위 안에서 model/method extension이 허용된다.

## 2. 공식 링크

- Official Task Page: https://noisy-text.github.io/2026/multi-lexnorm.html
- Baseline GitHub Repository: https://github.com/WeerayutBu/MultiLexNorm2026
- Submission Leaderboard: https://www.codabench.org/competitions/14162/?secret_key=33d4b8ec-4951-478b-8132-474e458409c3

## 3. 평가 정책

Leaderboard score는 중요한 참고 지표지만 유일한 평가 기준은 아니다. 과제 평가는 public leaderboard 점수만으로 결정되지 않는다.

중요한 추가 평가:

- 별도 cross-validation을 통해 generalization ability를 평가할 수 있음
- report에서 experimental setup, methodology, analysis를 명확히 설명해야 함
- 실험은 재현 가능해야 함

따라서 우리 작업에서는 public leaderboard 제출 기록보다 validation/cross-validation 설계, 실행 명령어, seed, split, dependency, 결과 해석을 문서화하는 것이 매우 중요하다.

## 4. 제출물

필수 제출물:

1. Report file
   - 파일명: `TeamName_Report`
2. Code files
   - 파일명: `TeamName_CodeFile.zip`
   - 모든 code files를 하나의 zip으로 압축
   - 필요시 `requirements.txt` 포함 필수
3. Leaderboard account email
   - 팀당 하나의 leaderboard 계정
   - CodaBench 등록 시 university email(`g.skku.edu`) 사용 권장/요구
4. Team work report
   - 개인별 제출
   - 최소 세 문장 이상
   - 본인 기여와 팀원별 기여를 구체적으로 작성
   - task, deliverable, approximate dates 포함
   - 팀원끼리 공유하거나 상의하면 안 됨
   - vague/insufficient report는 participation component에서 0점 가능

## 5. 보고서 작성 지침

보고서는 ACL template을 사용한다. 팀원 모두 SKKU Google account로 Overleaf 계정을 만들고, 팀장이 ACL template zip을 upload project로 생성한 뒤 팀원들을 Editor 권한으로 초대한다.

문헌 관리는 `custom.bib`에 BibTeX entry를 추가하고, main paper는 `acl_latex.tex`를 기반으로 작성한다.

보고서 분량:

- 5 pages

권장 reference:

- `UFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5`

보고서 구조:

1. Abstract
   - research objective, methodology, key findings 요약
2. Introduction
   - multilingual lexical normalization 문제 정의
   - motivation과 contribution 설명
3. Related Work
   - 기존 lexical normalization 접근 정리
   - multilingual 및 sequence-to-sequence 방법 논의
4. Model
   - ByT5 architecture 설명
   - fine-tuning strategy와 input/output format 설명
5. Experiments
   - dataset 설명
   - experimental setup과 evaluation metric 설명
6. Results
   - baseline 대비 성능 비교와 분석
7. Conclusion
   - study 요약
   - limitation 및 future research direction 제시

주의: Paper Writing Guideline에는 Experiments section에서 `MultiLexNorm 2021` dataset을 설명하라고 되어 있지만, Assignment Guideline의 실제 과제는 `MultiLexNorm2026`이다. 따라서 보고서에서는 2021을 related work/reference로 사용하고, 실제 실험 dataset은 2026 dataset으로 명확히 구분해야 한다.

## 6. 우리 작업에 반영할 운영 원칙

- baseline GitHub repository를 기반으로 시작한다.
- 새로운 실험 접근은 사용자에게 먼저 물어보고 진행한다.
- MFR은 개선용이 아니라 baseline 재현 및 한계 분석용으로 먼저 수행한다.
- public leaderboard 점수만 강조하지 않고 validation/cross-validation 결과를 반드시 남긴다.
- 보고서에는 `TBD`를 유지하되, 실험이 끝난 항목부터 실제 숫자와 명령어로 대체한다.
- 최종 제출 전에 `requirements.txt`, code zip, leaderboard email, 개인별 teamwork report 체크리스트를 확인한다.

## 7. 즉시 해야 할 일 후보

다음 단계는 사용자 선택 후 진행한다.

1. baseline GitHub repository를 가져와 코드 구조 확인
2. Hugging Face dataset 접근 권한/로그인 확인
3. `dev-pub` 기준 data loading만 확인
4. MFR baseline 실행 계획 작성
5. 보고서 `Model` section에 ByT5 설명을 더 자세히 보강
