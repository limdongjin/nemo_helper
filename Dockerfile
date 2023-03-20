##############################
FROM python:3.9.16-slim-buster as production

WORKDIR /app

###

RUN apt update && apt-get update -y && apt-get install -y libsndfile1 ffmpeg python-dev dumb-init 
RUN apt install -y git gcc g++

RUN pip install --upgrade pip && pip install poetry 
COPY pyproject.toml poetry.toml README.md ./

COPY model ./model
COPY examples ./examples
COPY nemo_helper ./nemo_helper

RUN python -m venv .venv
RUN /app/.venv/bin/python3.9 -m pip install --upgrade pip
RUN /app/.venv/bin/python3.9 -m pip install -e .
RUN /app/.venv/bin/python3.9 -m pip install Cython

###
RUN /app/.venv/bin/python3.9 -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
RUN git clone https://github.com/NVIDIA/NeMo
RUN /app/.venv/bin/python3.9 -m pip install -r /app/NeMo/requirements/requirements.txt
RUN /app/.venv/bin/python3.9 -m pip install -r /app/NeMo/requirements/requirements_common.txt
RUN /app/.venv/bin/python3.9 -m pip install -r /app/NeMo/requirements/requirements_asr.txt
RUN /app/.venv/bin/python3.9 -m pip install -r /app/NeMo/requirements/requirements_lightning.txt
RUN /app/.venv/bin/python3.9 -m pip uninstall pytorch-lightning -y
RUN /app/.venv/bin/python3.9 -m pip install pytorch-lightning==1.9.2
WORKDIR /app/NeMo
RUN /app/.venv/bin/python3.9 -m pip install -e .

### 
WORKDIR /app
### for debug
COPY samples ./samples
ENTRYPOINT ["sleep", "1000000"]

# WRITE your entrypoint
#ENTRYPOINT ["/usr/bin/dumb-init", "--"]
#CMD ["/app/.venv/bin/python3.9", "-m", "sid.entry"]
