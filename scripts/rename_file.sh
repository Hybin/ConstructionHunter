#!/bin/bash

directory="/Users/hybin/Documents/Code/PycharmProjects/ConstructionHunter/data/test"


for file in `ls ${directory}`; do
    # insert the prefix [test] for each file
    # mv ${directory}"/"$file ${directory}"/test_"$file
    left=${file%"，"*}
    right=${file##*"，"}

    if [[  ${left} != ${right} ]]; then
        filename=${left}"+，+"${right}
        mv ${directory}"/"${file} ${directory}"/"${filename}
    fi

done

ls ${directory}