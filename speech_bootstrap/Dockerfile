#FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /local

RUN pip install Cython
RUN apt update
RUN apt install -y git libsndfile1
RUN apt install -y gcc
RUN pip install pillow==4.1.1
RUN pip install llvmlite --ignore-installed
RUN git clone https://github.com/Edresson/Coqui-TTS -b multilingual-torchaudio-SE /local/TTS
RUN pip install -q -e /local/TTS/
RUN pip install matplotlib jupyter ipython

RUN mkdir /local/smart_drones
COPY requirements.txt /local/smart_drones
RUN pip install -r /local/smart_drones/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/local/smart_drones:/local"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y vim ffmpeg
RUN pip install -q pydub ffmpeg-normalize

COPY content.tar.gz /
RUN tar -xvf /content.tar.gz
RUN rm -rf /content.tar.gz
