printenv 

DIR=`pwd`/training

pwd
ls $DIR

export ELASTICSEARCH_URL=http://165.227.103.185:5601/api/console/proxy?path=
export ELASTICSEARCH_USER=
export ELASTICSEARCH_PASS=
export ELASTICSEARCH_INDEX=experiments2
export ELASTICSEARCH_INDEX_TYPE=measure
export GIT_SHA=https://github.com/garysieling/deeplens-experiments/commit/$(git rev-parse HEAD)
