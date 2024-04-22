FROM python:3.10
WORKDIR /code
COPY requirements.txt .
ENV PYTHONUNBUFFERED=1
RUN pip install -r requirements.txt
COPY unzip.py .
CMD [ "python", "./unzip.py" ]