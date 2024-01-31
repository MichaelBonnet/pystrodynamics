# Dockerfile
FROM python:3.10-bookworm as install
WORKDIR /app

# Update package manager and install python and pip
RUN apt-get update && apt-get install -y \
python3.10 \
python3-pip 

# Copy the requirements file over to the work directory and install all requirements
COPY requirements.txt .
RUN pip install --use-pep517 -r requirements.txt

COPY . .