#!/bin/bash

GITHUB_CDN="https://rawgit.com";

PROTOTXT="$GITHUB_CDN/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt"
CAFFEMODEL="$GITHUB_CDN/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"


# Create a pycaffe command that uses docker under the hood
pycaffe () { docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu ipython $@; }

# change to the current test directory
cd "$(dirname $0)"

WD="model"
ACT="activations"
PT="net.prototxt"
CM="net.caffemodel"

mkdir -p $WD
mkdir -p $ACT

# wget $PROTOTXT -O "$WD/$PT"
# wget $CAFFEMODEL -O "$WD/$CM"

pycaffe "inference.py"
