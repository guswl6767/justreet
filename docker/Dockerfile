FROM python:3.8-slim-buster
ENV DEBCONF_NOWARNINGS yes

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    git

COPY ./data/ /Projects
WORKDIR /Projects

# build from yolov5 source
RUN git clone https://github.com/ultralytics/yolov5.git
RUN pip3 install -r requirements.txt
