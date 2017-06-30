#!/bin/bash

for k in 2 10
do
    for l in 1 2 3 4 
    do
        fname="chicago_comp${k}_lvl${l}"
        echo $fname
        pdfcrop --bbox '90 60 650 380' $fname".pdf" $fname"_crop".pdf
    done

done

