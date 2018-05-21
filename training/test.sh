DIR=`pwd`

pwd
ls $DIR

# /var/jenkins_home/workspace/FastTraining/training

CMD="docker run --rm -e PASSWORD="pass" -it -v jenkins_home:/var/jenkins_home --mount type=bind,source=/data,target=/data segaleran/opencv-jupyter python $DIR/test.py $1"
echo $CMD
`$CMD`