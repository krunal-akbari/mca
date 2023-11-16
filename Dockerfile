#FROM ubuntu AS base
# RUN apt update && apt-get -y upgrade && \
#     apt-get install -y python3 pip  && \
#     apt-get install -y libgl1-mesa-glx && \
#     rm  -r /var/cache/apt/*
# RUN adduser kishan
# User kishan
# WORKDIR /home/kishan/
# COPY req.txt .
# RUN pip install -r req.txt
#
# COPY . .

FROM docker.io/continuumio/miniconda3 AS base
RUN apt-get update &&\
    apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y gdrive
COPY . .
RUN conda env create --name first --file=test.yml
RUN pip install --upgrade pip \
                pip install matplotlib \
                seaborn\
                scikit-learn \
                tensorflow \
                opencv-python
