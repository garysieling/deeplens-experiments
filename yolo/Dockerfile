FROM ubuntu:16.04
RUN mkdir /darknet
WORKDIR /darknet
RUN \
	apt-get update && apt-get install -y \
	autoconf \
        automake \
	libtool \
	build-essential \
	git

RUN \
	apt-get install -y \
	wget

RUN \
	git clone https://github.com/pjreddie/darknet && \
	cd darknet && \
	make

RUN \ 
	wget http://pjreddie.com/media/files/yolo.weights >/dev/null 2>&1 && \
	wget http://pjreddie.com/media/files/tiny-yolo-voc.weights >/dev/null 2>&1
	
RUN \ 
	wget http://pjreddie.com/media/files/vgg-conv.weights >/dev/null 2>&1

RUN \
	wget http://pjreddie.com/media/files/shakespeare.weights >/dev/null 2>&1 && \
	wget http://pjreddie.com/media/files/grrm.weights >/dev/null 2>&1 && \
	wget http://pjreddie.com/media/files/tolstoy.weights >/dev/null 2>&1 && \
	wget http://pjreddie.com/media/files/kant.weights >/dev/null 2>&1

RUN \
	wget https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt >/dev/null 2>&1

RUN \
	wget pjreddie.com/media/files/go.weights >/dev/null 2>&1

RUN \	
	cd darknet/ && \
	./darknet && \
	./darknet detector test cfg/coco.data cfg/yolo.cfg /root/yolo.weights data/dog.jpg && \
	./darknet rnn generate cfg/rnn.cfg /root/shakespeare.weights -srand 0 -seed CLEOPATRA -len 300

CMD ["bash"]