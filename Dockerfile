FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY . .
COPY model ./model

# COPY app.py .

EXPOSE 2000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "2000"]