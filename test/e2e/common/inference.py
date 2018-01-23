import sys
import caffe
from PIL import Image
import numpy as np
import os
import argparse


def main():
    args = parse_args()
    data = load_img(url=args.url, size=args.size)
    data = preprocess_input(data, mean=args.mean)
    net, out = predict(data)
    save_blobs(net, out)
    print topk(out)

def parse_args():
    """Parse CLI options"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", dest="proto", help="path to prototxt file")
    parser.add_argument("--model", dest="model", help="path to caffemodel file")
    parser.add_argument("--image", dest="url", help="path to the input image")
    parser.add_argument("--size", dest="size", help="path to caffemodel file",
                        default=[224, 224], type=int, nargs='+')
    parser.add_argument("--mean", dest="mean", help="path to caffemodel file",
                        default=[104, 117, 123], type=int, nargs='+',)

    return parser.parse_args()

def load_img(url="cat.jpg", size=(224, 224)):
    """Read image data and transform colormode RGB to BGR"""
    im = Image.open(url)
    im = im.resize(size, Image.ANTIALIAS)
    return np.array(im)[:,:,::-1].astype('float32')

def preprocess_input(data, mean=[104, 117, 123], dims=[0,3,2,1]):
    """Preprocess input image"""
    # Subtract training mean
    data = data - np.array(mean)

    # Reshape to Caffe input
    return np.expand_dims(data, axis=0).transpose(*dims)

def predict(data, proto="model/net.prototxt", model="model/net.caffemodel", all_blobs=True):
    """Load a Caffemodel and predict"""
    caffe.set_mode_cpu()
    net = caffe.Net(proto, model, caffe.TEST)
    blobs = net.blobs.keys() if all_blobs else []
    out = net.forward_all(blobs=blobs, data=data)
    return net, out

def reshape(blob, dims=[2,1,0]):
    """Reshape a blob to deeplearn.js"""
    arr = np.squeeze(blob, axis=0)
    if len(arr.shape) is 3:
        arr = arr.transpose(*dims)
    return arr

def mkdir(filename):
    """Create a parent directory if not exists"""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def write(filename, blob):
    """Dump blob to disk as flat binary array"""
    mkdir(filename)
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    f.close()

def save_blobs(net, out):
    """Store all activations as flat binaries"""
    for blob in net.blobs.keys():
        data = reshape(out[blob]).astype('float32')
        write('activations/%s' % blob, data)

def topk(out):
    """Output Top-5 values and index"""
    prob = out['prob'].flatten()
    indices = np.argsort(-prob.flatten())[:5]
    return indices, prob[indices]

if __name__ == '__main__':
    main()
