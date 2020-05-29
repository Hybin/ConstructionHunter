#!/bin/bash

pid='/Users/hybin/Documents/Code/PycharmProjects/ConstructionHunter/log/CoreNLP_pid.txt'
kill -9 `cat $pid`

echo "Stop the Server successfully!"
rm $pid