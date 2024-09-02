FROM python:3.9

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /opt/ml/code

COPY RFC_model.pkl .
COPY train.py .
COPY predictor.py .
COPY serve.py .

ENV SAGEMAKER_PROGRAM=predictor.py
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "/opt/ml/code/serve.py"]
CMD ["serve.py"]

