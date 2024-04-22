FROM python:3.10
WORKDIR /code
COPY requirements.txt .
COPY MarsDataset.py .
ENV PYTHONUNBUFFERED=1
RUN pip install -r requirements.txt
COPY unet.py .
CMD [ "python", "./unet.py" ]