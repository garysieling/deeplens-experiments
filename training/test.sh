DIR=`pwd`

pwd
ls $DIR

CMD="docker run --rm -e PASSWORD=\"pass\" type=bind,source=$DIR,target=/work --mount type=bind,source=/data,target=/data segaleran/opencv-jupyter ls /work"
echo $CMD
`$CMD`