#!/bin/bash
python main.py -m hog+svm;
python main.py -m sift+svm;
python main.py -m hog+kernel_svm;
python main.py -m sift+kernel_svm;
python main.py -m hog+kmeans;
python main.py -m sift+kmeans;
for transform in "" "randomflip" "randomcrop" "normalize"
do
    for net in "MLP" "Vgg" "Resnet" 
    do
        for optim in "Adam" "SGD" "Adagrad"
        do
            for reg in "" "-dropout" "-weightdecay"
            do
            python test.py -d $1 -n $net -optim $optim $reg -t $transform;
            done
        done
    done
done