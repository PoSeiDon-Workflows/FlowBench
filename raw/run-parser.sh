#!/usr/bin/env bash

echo "Untarring elasticsearch..."
#combine partial elasticsearch archives
cat archive/elasticsearch.tar.xz.* > elasticsearch.tar.xz

#untar elasticsearch archive
tar -xf elasticsearch.tar.xz

echo "Untarring workflow submit dirs..."
#combine partial workflow submit dirs archives
cat archive/workflow-submit-dirs.tar.xz.* > workflow-submit-dirs.tar.xz

#untar workflow submit dirs archive
tar -xf workflow-submit-dirs.tar.xz

#make sure that the elasticsearch folder is readable/writable
chmod -R 777 elasticsearch

#start up the elasticsearch container
docker-compose up -d

#sleep and wait for elasticsearch to come up
echo "Sleeping for 60 seconds..."
sleep 60

#create the python environment for the parser
echo "Creating python environment..."
python3 -m venv parser-pyenv

#enable the pythone environment
source parser-pyenv/bin/activate

#install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install pandas elasticsearch==6.2.0
python3 -m pip install --upgrade urllib3

#run the parser
echo "Starting parser..."
python3 parse-data.py

#deactivate the environment
deactivate

#stop and remove the elasticsearch container
docker-compose down
