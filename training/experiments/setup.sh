printenv 

cd training
chmod +x test.sh
DIR=`pwd`

pwd
ls $DIR

ELASTICSEARCH_URL=http://165.227.103.185:5601/api/console/proxy?path=
ELASTICSEARCH_USER=
ELASTICSEARCH_PASS=
ELASTICSEARCH_INDEX=experiments2
ELASTICSEARCH_INDEX_TYPE=measure
GIT_SHA=https://github.com/garysieling/deeplens-experiments/commit/$(git rev-parse HEAD)
