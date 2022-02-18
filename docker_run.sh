#/usr/bin/bash

docker run -it \
           -v ${PWD}:/local/smart_drones \
           --rm vocal:0.1 /bin/bash
#docker run -it \
           #-v ${PWD}:/local/smart_drones \
           #--gpus all \
           #--rm vocal:0.1 /bin/bash
