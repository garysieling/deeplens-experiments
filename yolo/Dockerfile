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
	wget -q https://pjreddie.com/media/files/yolov3.weights

RUN \	
	cd darknet/ && \
	./darknet 

CMD ["bash"]
