FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt