# ByT5 실험 환경 세팅

이 문서는 MultiLexNorm2026에서 ByT5 기반 실험을 시작하기 위한 환경 세팅 명령어를 정리한다. 실험 실행은 사용자가 직접 터미널에서 수행한다.

## 1. 현재 목표

우리는 UFAL MultiLexNorm 2021 시스템의 핵심 아이디어를 2026 데이터에 맞게 이식하려고 한다.

핵심 구조:

```text
input:  left context <extra_id_0> raw_token <extra_id_1> right context
output: norm_token
model:  google/byt5-small
```

처음부터 UFAL의 full synthetic pretraining까지 복제하지 않는다. 먼저 2026 authentic train data만 사용해서 context-marked ByT5 fine-tuning이 가능한 환경을 만든다.

## 2. 권장 환경 생성

### Option A. venv 사용

프로젝트 폴더에서 실행한다.

```bash
cd /home/raymond/new_project
python3 -m venv .venv-byt5
source .venv-byt5/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-byt5.txt
```

### Option B. conda 사용

CUDA/PyTorch 버전을 conda로 관리하고 싶으면 이 방식을 사용한다.

```bash
cd /home/raymond/new_project
conda create -n multilexnorm-byt5 python=3.10 -y
conda activate multilexnorm-byt5
python -m pip install --upgrade pip
pip install -r requirements-byt5.txt
```

## 3. GPU 확인

환경 설치 후 아래 명령어로 PyTorch와 GPU 인식 여부를 확인한다.

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

CUDA가 `False`여도 데이터 분석과 작은 smoke test는 가능하지만, `google/byt5-small` fine-tuning은 CPU에서 매우 느릴 수 있다.

## 4. Hugging Face 모델 다운로드 확인

ByT5 tokenizer/model을 불러올 수 있는지 확인한다.

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("tokenizer:", type(tokenizer).__name__)
print("model:", type(model).__name__)
print("parameters:", sum(p.numel() for p in model.parameters()))
PY
```

이 단계는 처음 실행할 때 Hugging Face에서 모델을 다운로드한다.

## 5. 데이터 파일 확인

이미 다운로드한 데이터가 있는지 확인한다.

```bash
find /home/raymond/new_project/data/raw -maxdepth 3 -type f | sort
```

정상이라면 다음 parquet 파일들이 보여야 한다.

```text
data/raw/multilexnorm2026-dev-pub/data/train-00000-of-00001.parquet
data/raw/multilexnorm2026-dev-pub/data/validation-00000-of-00001.parquet
data/raw/multilexnorm2026-dev-pub/data/test-00000-of-00001.parquet
data/raw/multilexnorm2026-full-pub/data/train-00000-of-00001.parquet
data/raw/multilexnorm2026-full-pub/data/validation-00000-of-00001.parquet
data/raw/multilexnorm2026-full-pub/data/test-00000-of-00001.parquet
```

## 6. 다음 단계

환경이 준비되면 바로 model fine-tuning으로 가지 않는다. 먼저 다음 순서로 작은 확인을 한다.

1. MFR validation 결과 재현
2. MFR 실패 token의 seen/unseen 분석
3. ByT5 학습 데이터 builder 작성
4. 아주 작은 subset으로 ByT5 forward/generation smoke test
5. 그 다음에 fine-tuning 실행 여부 결정

이 순서를 지키는 이유는 pretrained model이 실제로 MFR의 어떤 한계를 보완하는지 보고서에서 설명하기 위해서다.

## 7. 설치 문제가 생기면

오류가 나면 아래 정보를 함께 기록한다.

```bash
python --version
pip --version
pip freeze | grep -E 'torch|transformers|datasets|pyarrow|polars|accelerate'
nvidia-smi
```

오류 메시지를 그대로 공유하면 dependency 버전 또는 CUDA 문제를 기준으로 다음 조치를 정할 수 있다.
