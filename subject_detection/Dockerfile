FROM tensorflow/tensorflow:latest-gpu

WORKDIR /local

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -yq --no-install-recommends
RUN apt-get install -y libsndfile1-dev git wget
RUN pip3 install -q git+https://github.com/vasudevgupta7/gsoc-wav2vec2@main

RUN pip3 install -q SoundFile
RUN pip3 install -q ipython
