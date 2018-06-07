docker run -v /data/frames:/frames -v $(pwd):/source -it --rm mxnet/python:1.2.0_cpu python /source/novelty.py 256 256
