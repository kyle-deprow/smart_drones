FROM pytorch/pytorch:latest

WORKDIR /local

RUN pip3 install Cython
RUN apt update
RUN apt install -y git
RUN git clone https://github.com/Edresson/Coqui-TTS -b multilingual-torchaudio-SE /local/TTS
RUN apt install -y gcc
RUN pip install -q -e /local/TTS/
RUN pip3 install matplotlib jupyter ipython

RUN mkdir /local/smart_drones
COPY requirements.txt /local/smart_drones
RUN pip3 install -r /local/smart_drones/requirements.txt

