FROM python:3.11-slim

WORKDIR /app

# Install system dependencies spaCy needs on Linux
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model — this is the one blocked on Windows
# Inside this Linux container it installs perfectly
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
# Download NLTK data (RAKE needs stopwords)
RUN python -c "import nltk; \
    nltk.download('stopwords'); \
    nltk.download('punkt'); \
    nltk.download('punkt_tab')"

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]