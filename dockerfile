FROM python:3.8-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the training script
COPY train.py .

# Run the training script
ENTRYPOINT ["python", "train.py"]
