# MFR Error Analysis Plan

목적: MFR baseline이 왜 pretrained ByT5 실험으로 이어지는지 실험적으로 보여준다.

## 분석 질문

MFR은 train data에서 본 raw token에 대해서만 가장 빈번한 normalization을 적용한다. 따라서 validation에서 틀린 token이 train에 없던 unseen raw token인지, 아니면 train에 있었지만 잘못된 most-frequent replacement를 선택한 seen raw token인지 구분해야 한다.

이 분석은 다음 주장을 뒷받침한다.

> MFR is a memorization baseline. Its errors on unseen noisy forms motivate byte-level pretrained generation with ByT5.

## 스크립트

- `scripts/mfr/analyze_mfr_errors.py`

기능:

- `data/raw/multilexnorm2026-dev-pub`의 train/validation parquet 로딩
- train split에서 MFR dictionary 생성
- validation split 예측
- LAI accuracy, MFR accuracy, ERR 계산
- 언어별 ERR 계산
- MFR error를 `seen_error`와 `unseen_error`로 분류
- changed token을 `seen_changed`와 `unseen_changed`로 분류
- report에 넣을 예시 오류 저장

## 실행 명령어

프로젝트 루트에서 실행한다.

```bash
cd /home/raymond/new_project
source .venv-byt5/bin/activate  # venv를 쓴 경우
# 또는 conda activate multilexnorm-byt5

python scripts/mfr/analyze_mfr_errors.py
```

출력 파일:

- `outputs/mfr/analysis/mfr_validation_error_analysis.json`
- `outputs/mfr/analysis/mfr_validation_error_analysis.md`

## 결과 해석 방법

중요하게 볼 항목:

- `unseen_error` 비율이 높으면 MFR이 memorization 한계에 많이 걸린다는 뜻이다.
- `seen_error` 비율이 높으면 같은 raw token의 ambiguous normalization 또는 most-frequent 선택 오류가 많다는 뜻이다.
- `unseen_changed` 비율이 높으면 validation의 실제 normalization 대상 중 train에서 본 적 없는 noisy form이 많다는 뜻이다.

보고서 연결:

- unseen error가 많으면 ByT5의 byte-level generalization 필요성을 강조한다.
- seen error가 많으면 문맥 정보가 필요한 경우가 있을 수 있으므로 UFAL식 context-marked input의 필요성을 강조한다.
