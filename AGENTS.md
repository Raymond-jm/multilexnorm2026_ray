# Project Guidance for Codex

이 파일은 MultiLexNorm2026 과제를 진행할 때 항상 참고할 작업 기준이다. 보고서와 실험은 아래 가이드라인을 우선으로 맞춘다.

## Assignment Summary

- 과제명: MultiLexNorm2026, Multilingual Lexical Normalization
- 목표: SNS/온라인 커뮤니티 등의 noisy text에서 비표준 어휘를 표준형으로 정규화한다.
- 공식 평가 지표: ERR(Error Reduction Rate)
- 제공 baseline: MFR(Most-Frequent-Replacement), data loading, evaluation utilities
- 제출물:
  - `TeamName_Report`
  - `TeamName_CodeFile.zip`
  - `requirements.txt` 포함
  - leaderboard 계정 이메일
  - 개인별 team work report

## Important Evaluation Notes

- leaderboard 점수만으로 평가되지 않는다.
- 별도 cross-validation/generalization 평가가 있을 수 있다.
- 보고서에는 실험 설정, 방법론, 결과 분석, 재현 가능성이 명확히 드러나야 한다.
- public leaderboard에만 과적합하지 않도록 validation/cross-validation 실험을 기록한다.

## Official Task Notes

- task는 17개 언어를 포함한다.
- final ranking은 언어별 ERR의 macro-average 중심이다.
- 공식 페이지에서 단일 모델만 사용해야 한다는 제한은 확인되지 않았다.
- 따라서 하나의 제출 시스템 안에서 다음 접근을 비교할 수 있다:
  - 전체 언어를 하나로 학습한 multilingual model
  - 언어별 dictionary/model을 사용하는 routed system
  - MFR + rules + neural model ensemble
- 단, 최종 submission은 모든 언어에 대해 정해진 format의 prediction을 생성해야 한다.
- 추가 pre-trained model은 사용할 수 있다.
- target language lexicon-normalization 추가 데이터 사용은 제한이 있으므로 주의한다.

## Report Requirements

- ACL template 기반으로 작성한다.
- 보고서 분량 목표: 5 pages
- 한국어 초안은 `docs/report_draft_ko.md`에 작성한다.
- 나중에 최종 제출 전 영어 ACL paper 형태로 옮긴다.
- 참고 논문:
  - `ÚFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5`
  - `ByT5: Towards a token-free future with pre-trained byte-to-byte models`

## Report Structure

1. Abstract
2. Introduction
3. Related Work
4. Model
5. Experiments
6. Results
7. Conclusion

## Experiment Policy

- 사용자가 직접 터미널에서 실험을 실행한다.
- Codex는 실험 설계, 실행 명령어, 코드 초안, 결과 해석, 보고서 문장화를 돕는다.
- 실험 결과는 `experiments/experiment_log.md`에 계속 기록한다.
- 진행 상황은 `docs/progress.md`에 기록한다.
- Codex는 사용자의 승인 없이 모델 학습, 평가, leaderboard 제출, 대규모 다운로드를 진행하지 않는다.
- Codex가 스크립트를 만들 때는 사용자가 이해할 수 있도록 핵심 로직에 설명 주석을 넣는다.
- Codex가 제공하는 terminal command는 사용자가 직접 실행할 수 있도록 project root 기준으로 작성한다.
- 사용자가 실행한 terminal output을 받으면, 해당 숫자와 command를 그대로 실험 로그와 보고서 초안에 반영한다.

## Experiment Organization

- 실험 관련 script는 방법별 하위 폴더에 둔다.
  - MFR: `scripts/mfr/`
  - ByT5: `scripts/byt5/`
  - 기타 방법: `scripts/<method_name>/`
- 실험 관련 문서는 방법별 하위 폴더에 둔다.
  - MFR: `docs/experiments/mfr/`
  - ByT5: `docs/experiments/byt5/`
  - 기타 방법: `docs/experiments/<method_name>/`
- 실험 출력물은 방법별 하위 폴더에 둔다.
  - MFR: `outputs/mfr/`
  - ByT5: `outputs/byt5/`
  - 기타 방법: `outputs/<method_name>/`
- 학습 입력으로 생성한 sample/intermediate data는 `outputs/`가 아니라 `sample_data/`에 둔다.
  - ByT5: `sample_data/byt5/`
- 공통 진행 기록은 `docs/progress.md`에 남긴다.
- 전체 실험 로그는 `experiments/experiment_log.md`에 누적한다.

## Result Recording Rules

- 보고서에 들어가는 수치는 실제 실행 결과만 사용한다.
- 숫자를 추측하거나 논문 수치와 우리 실험 수치를 섞어 쓰지 않는다.
- 논문에서 가져온 수치는 반드시 source를 명시하고, 우리 실험 결과와 분리해서 기록한다.
- 실험 결과를 기록할 때는 최소한 다음 정보를 함께 남긴다.
  - 실행 날짜
  - 실행자
  - 실행 command
  - dataset 이름과 split
  - 모델/방법 이름
  - 주요 옵션 또는 hyperparameter
  - metric 결과
  - 출력 파일 경로
- validation 결과, test submission 결과, public leaderboard 결과, cross-validation 결과를 서로 구분해서 기록한다.
- `full-pub`와 `dev-pub`의 차이를 명확히 기록한다. 현재 확인 기준으로 train/validation은 동일하고 test만 다르다.
- validation split에 없는 언어(`da`, `es`, `it`, `tr`, `trde`)가 있으므로 validation 결과를 17개 언어 전체 성능처럼 과장하지 않는다.

## Current Strategy

1. Baseline repository setup
2. Conda environment setup
3. Hugging Face authentication 확인
4. Dataset loading 확인
5. MFR baseline 재현
6. 언어별 MFR 한계 분석

## Experiment Planning Rule

- 새로운 실험 접근법을 계획에 넣기 전 반드시 사용자에게 먼저 묻는다.
- 새로운 실험을 하기 전에는 전과 같은 옵션을 사용하여 실험하고 옵션을 바꿔야 한다면 먼저 물어본다.
- MFR은 성능 개선 목적이 아니라 baseline 재현과 한계 분석 목적으로 사용한다.
- 일단 브레인스토밍으로 seq2seq문제를 풀 수 있는 최신 기법들을 최대한 조사한다.
- ERR이 더 나아지지 않을 거 같으면 그 쪽 브랜치는 그만 진행하고 처음 브랜치에서 다른방향으로 다시 진행한다.

## Writing Style

- 사용자가 이해하고 직접 보고서를 쓸 수 있도록 한국어 초안으로 먼저 작성한다.
- 실험 결과가 없는 부분은 `TBD`로 둔다.
- 숫자, 명령어, 데이터 split, 모델 설정은 추측하지 않고 실제 실험 결과를 기록한다.
