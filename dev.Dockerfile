FROM python:3.11-slim

WORKDIR /app

COPY requirements-freeze.txt ./
RUN pip install --no-cache-dir -r requirements-freeze.txt

COPY . .

RUN pytest -q || true

CMD ["bash"]
