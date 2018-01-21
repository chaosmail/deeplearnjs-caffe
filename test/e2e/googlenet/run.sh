#!/bin/bash

GITHUB_CDN="https://rawgit.com";
BERKELEY_DL="http://dl.caffe.berkeleyvision.org"

PROTOTXT="$GITHUB_CDN/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt"
CAFFEMODEL="$BERKELEY_DL/bvlc_googlenet.caffemodel"

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

wget $PROTOTXT -O "$WD/$PT"
wget $CAFFEMODEL -O "$WD/$CM"

pycaffe "inference.py"
