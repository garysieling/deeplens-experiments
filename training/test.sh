cd training
chmod +x test.sh
DIR=`pwd`

pwd
ls $DIR

docker run --rm -e PASSWORD="pass" -t -v jenkins_home:/var/jenkins_home --mount type=bind,source=/data,target=/data segaleran/opencv-jupyter python $DIR/test.py 4