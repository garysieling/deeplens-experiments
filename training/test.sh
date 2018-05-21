DIR=printf "%q" `pwd`

pwd
ls /home/jovyan/work

CMD="docker run --rm -e PASSWORD=\"pass\" -v $DIR:/home/jovyan/work --mount type=bind,source=/data,target=/data segaleran/opencv-jupyter python /home/jovyan/work/test.py"
echo $CMD
`$CMD`