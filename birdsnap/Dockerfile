FROM python:2-stretch
RUN mkdir dataset
WORKDIR dataset
RUN curl -o birdsnap.tgz http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz
RUN tar -xvf birdsnap.tgz
WORKDIR /dataset/birdsnap
RUN pip install requests
RUN python get_birdsnap.py
ENTRYPOINT [ "bash" ]