FROM python:3.12

ENV WORKSPACE=/workspace

WORKDIR $WORKSPACE

COPY requirements.txt .

RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

RUN huggingface-cli download TrustSafeAI/RADAR-Vicuna-7B

COPY . .

EXPOSE 8000

ENTRYPOINT uvicorn main:app --log-config=log_conf.yml --host 0.0.0.0 --workers 1


