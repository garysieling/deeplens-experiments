cd jenkins
docker build --tag garysieling/jenkins .

docker run \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v jenkins_home:/var/jenkins_home \
  --name jenkins \
  garysieling/jenkins 
