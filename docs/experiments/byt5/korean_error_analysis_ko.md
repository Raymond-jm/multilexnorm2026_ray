# Korean-specific ByT5 Error Analysis

대상 실험:

- Model: `outputs/byt5/lang_ko_1000steps/checkpoint`
- Predictions: `outputs/byt5/lang_ko_1000steps/validation_eval_ko/predictions.jsonl`
- Validation: Korean-only validation

## 요약

한국어 전용 ByT5 1000-step 모델은 validation에서 MFR보다 낮은 성능을 보였다.

```text
LAI accuracy: 91.17
Model accuracy: 79.41
ERR: -133.13
```

오류 분석 결과, 핵심 문제는 두 가지다.

1. `changed` token을 하나도 정확히 맞추지 못했다.
2. `unchanged` token 중 221개를 불필요하게 바꿨다.

## Bucket 분석

| Bucket | Count | Percent |
| --- | ---: | ---: |
| unchanged_correct | 1,493 | 79.41 |
| unchanged_overedited | 221 | 11.76 |
| changed_copied_raw | 156 | 8.30 |
| changed_wrong_other | 10 | 0.53 |

Changed token 기준:

```text
changed total: 166
changed correct: 0
changed accuracy: 0.00%
```

Unchanged token 기준:

```text
unchanged total: 1714
unchanged correct: 1493
unchanged accuracy: 87.11%
```

## 해석

모델은 대부분의 changed token에서 raw token을 그대로 복사했다. 예:

```text
애니프사 -> 애니메이션프사, pred=애니프사
여친이랑 -> 여자친구랑, pred=여친이랑
띵작이다 -> 명작이다, pred=띵작이다
ㄹㅇ -> 진짜, pred=ㄹㅇ
```

동시에 unchanged token도 일부 과도하게 바꿨다. 예:

```text
로 -> 진짜
잘생겼는데 -> 진짜
지금 -> 다만
10년 -> 년
```

즉 이 모델은 changed token을 정규화하는 능력을 얻지 못했고, 일부 context에서는 train에서 본 target token을 환각처럼 출력했다.

## 가능한 원인

- 한국어 train에서 changed token은 958개뿐이라 supervised signal이 작다.
- 전체 token 학습에서는 unchanged token이 압도적으로 많아 copy behavior가 강하게 학습된다.
- 1000 step은 한국어 전체 token 기준 약 0.3 epoch 정도라 changed pattern을 충분히 학습하지 못했을 수 있다.
- 반대로 특정 target token을 memorization하여 unchanged token에 과도하게 출력하는 현상도 보인다.
- 한국어 lexical normalization에는 의미 확장형이 많다. 예: `여친 -> 여자친구`, `ㄹㅇ -> 진짜`, `띵작 -> 명작`. 단순 철자 보정이 아니라 slang expansion이라 ByT5가 더 많은 changed examples를 필요로 할 수 있다.

## 다음 실험 후보

1. `changed + unchanged downsampling`
   - changed token은 전부 사용
   - unchanged token은 changed의 1-3배만 샘플링
   - 목적: changed signal 강화와 over-copy 완화
2. 학습 step 증가
   - 1000 step은 짧을 수 있음
   - 단, over-editing이 심해질 수 있으므로 validation을 자주 확인해야 함
3. MFR fallback hybrid
   - model prediction이 raw와 크게 다르거나 confidence가 낮으면 MFR/raw copy 사용
4. changed-only model은 위험
   - unchanged token을 그대로 두는 능력이 약해질 수 있으므로 단독 사용은 추천하지 않음
