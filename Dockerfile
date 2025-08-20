
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py /app/bot.py
COPY .env /app/.env  # create from .env.example
RUN mkdir -p /app/logs

CMD ["python", "-u", "bot.py"]
