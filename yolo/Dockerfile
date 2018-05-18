FROM alpine
RUN apk update && \
    apk upgrade && \
    apk add make git
RUN mkdir /darknet
WORKDIR /darknet
RUN git clone https://github.com/pjreddie/darknet
RUN apk add gcc
RUN apk add libc-dev
WORKDIR /darknet/darknet
RUN make
RUN wget https://pjreddie.com/media/files/yolov3.weights
ENTRYPOINT [ "bash" ]