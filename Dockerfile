FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static
COPY sample_docs ./sample_docs
ENV PORT=7860 PUBLIC_DEMO=1 MAX_UPLOAD_MB=2 MAX_DOCS=15 DATA_DIR=/tmp/data
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','7860')+'/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
