printenv 

cd training
chmod +x test.sh
DIR=`pwd`

pwd
ls $DIR

docker run --rm -e PASSWORD="pass" -t -v jenkins_home:/var/jenkins_home --mount type=bind,source=/data,target=/data -e "BUILD_NUMBER=$BUILD_NUMBER" -e "BUILD_NAME=$JOB_NAME" segaleran/opencv-jupyter python $DIR/test.py 4