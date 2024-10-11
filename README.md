Test whether GPU works in container environment.

## Run

### Manually

#### Activate a virtual envoronment

#### Install dependence

```bash
pip install -r requirements.txt
```

#### Start up

```bash
uvicorn main:app --log-config=log_conf.yml --host 0.0.0.0 --workers 1
```

Tips:
- Type Ctrl-Z to suspend the current process.
- Type `jobs` to list jobs.
- Type `fg` to move suspended jobs to the frontground,
  or type `bg` to move suspended jobs to the background.

### Via Docker

#### Build image

```bash
docker build -t gpu-test .
```

#### Start a container

```bash
docker run -d --gpus '"device=3"' -p 8000:8000 -e BATCH_SIZE_MAX=8 -e BATCH_TIMEOUT=0.5 gpu-test
```

Run the following command to test:

```bash
curl -w @curl-format -o /dev/null -s -X 'POST' \
    'http://127.0.0.1:8000/predit/text' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "document": "Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.",
    "version": "no-version",
    "multilingual": false
}'
```

Run the following command to enter the container:

```bash
docker exec -it gpu-test /bin/bash
```

#### Stop & remove a container

```bash
docker stop gpu-test
docker rm gpu-test
```

## Performance test

Use [locust](https://docs.locust.io/en/stable/) on remote machine (client).

```bash
locust -f test.py -H http://34.67.65.126:8000
```

