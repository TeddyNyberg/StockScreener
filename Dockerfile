
FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY app/ .

CMD ["python", "main.py"]