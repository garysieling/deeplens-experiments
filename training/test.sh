docker run --rm -e PASSWORD="pass" -v /home/admin:/home/jovyan/work --mount type=bind,source=/data,target=/data segaleran/op
encv-jupyter python test.py