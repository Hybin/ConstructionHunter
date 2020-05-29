#!/bin/bash

root='/Users/hybin/Documents/Code/PycharmProjects/ConstructionHunter//libraries/stanford-corenlp/'
log='/Users/hybin/Documents/Code/PycharmProjects/ConstructionHunter//log/CoreNLP_log.txt'
pid='/Users/hybin/Documents/Code/PycharmProjects/ConstructionHunter//log/CoreNLP_pid.txt'

# Go to the directory of CoreNLP Server
cd $root

# Start the Server
nohup java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 8088 -timeout 15000 > $log 2>&1 &

# Record the pid
touch $pid
echo $! > $pid
echo "start the server successfully!"


