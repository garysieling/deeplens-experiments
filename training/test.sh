DIR=`pwd`

pwd
ls $DIR

# /var/jenkins_home/workspace/FastTraining/training

CMD="docker run --rm -e PASSWORD="pass" -v jenkins_home:/var/jenkins_home --mount type=bind,source=/data,target=/data segaleran/opencv-jupyter python $DIR/test.py"
echo $CMD
`$CMD`