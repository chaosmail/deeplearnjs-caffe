#!/bin/bash

GITHUB_CDN="https://rawgit.com";
BERKELEY_DL="http://dl.caffe.berkeleyvision.org"

PROTOTXT="$GITHUB_CDN/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt"
CAFFEMODEL="$BERKELEY_DL/bvlc_googlenet.caffemodel"

# Create a pycaffe command that uses docker under the hood
pycaffe () {
  docker run --rm -u $(id -u):$(id -g) \
    -v $(pwd):$(pwd) \
    -v $(dirname $(pwd))/assets:$(pwd)/assets \
    -v $(dirname $(pwd))/common:$(pwd)/common \
    -w $(pwd) bvlc/caffe:cpu ipython $@;
}

# change to the current test directory
cd "$(dirname $0)"

WD="model"
PT="net.prototxt"
CM="net.caffemodel"
mkdir -p $WD

ACT="activations"
mkdir -p $ACT

wget $PROTOTXT -O "$WD/$PT"
wget $CAFFEMODEL -O "$WD/$CM"

pycaffe "common/inference.py" -- \
  --image "$(pwd)/assets/cat_224x224.jpg" --proto "$WD/$PT" --model "$WD/$CM" \
  --size 224 224 --mean 104 117 123
