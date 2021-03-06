FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('rslp')" ]

COPY . /app

#Maybe you'll have to change the timeout parameter when de dataset grows up
CMD ["gunicorn", "--workers=2","--threads=2","--timeout=360", "--worker-class=gthread", "--bind", "0.0.0.0:5000", "app:app"]