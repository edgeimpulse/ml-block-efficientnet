#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb0_notop.h5
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb1_notop.h5
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb2_notop.h5
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb3_notop.h5
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb4_notop.h5
wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb5_notop.h5
