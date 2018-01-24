#!/bin/bash

for i in "$@"
do
case $i in
    --fetch)
    FETCH=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

GITHUB_CDN="https://rawgit.com";

PROTOTXT="$GITHUB_CDN/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt"
CAFFEMODEL="$GITHUB_CDN/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"

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
ACT="activations"
PT="net.prototxt"
CM="net.caffemodel"

mkdir -p $WD
mkdir -p $ACT

if [ "$FETCH" = "YES" ]; then
  wget $PROTOTXT -O "$WD/$PT"
  wget $CAFFEMODEL -O "$WD/$CM"
fi

pycaffe "common/inference.py" -- \
  --image "$(pwd)/assets/cat_227x227.jpg" --proto "$WD/$PT" --model "$WD/$CM" \
  --size 227 227 --mean 104 117 123
