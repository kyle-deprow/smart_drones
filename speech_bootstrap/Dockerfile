#FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /local

RUN apt update
RUN apt install -y git libsndfile1 gcc vim
RUN apt-get install -y python3-pip python3-venv

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel

# Create venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Visualization tools for python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libjpeg-dev zlib1g-dev ffmpeg
RUN pip3 install matplotlib ipython jupyter


# Pip install
RUN mkdir /local/speech_bootstrap
RUN mkdir /local/speech_bootstrap/src/
COPY ./src/  /local/speech_bootstrap/src
RUN ls /local/speech_bootstrap/
COPY ./requirements.txt  /local/speech_bootstrap/
COPY ./setup.py  /local/speech_bootstrap/
RUN python -m pip install -e /local/speech_bootstrap/
ADD ./docker/punkt_install.py   /local/
RUN python /local/punkt_install.py
RUN rm /local/punkt_install.py

RUN git clone https://github.com/Edresson/Coqui-TTS -b multilingual-torchaudio-SE /local/TTS
RUN pip install -q -e /local/TTS/

COPY content.tar.gz /
RUN tar -xvf /content.tar.gz
RUN rm -rf /content.tar.gz
