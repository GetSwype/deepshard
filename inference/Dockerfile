FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y git

RUN pip3 install torch accelerate sentencepiece huggingface_hub
RUN pip3 install git+https://github.com/GetSwype/transformers

COPY cli.py /app/cli.py

ENV HUGGINGFACE_HUB_CACHE=/app/hf-data

# ENTRYPOINT ["python3", "/app/cli.py", "--model_name=swype/deepshard-13B-ft"]