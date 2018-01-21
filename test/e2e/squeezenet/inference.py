import sys
import caffe
from PIL import Image
import numpy as np
import os

image_path = "cat.jpg"

caffe.set_mode_cpu()
net = caffe.Net("model/net.prototxt", "model/net.caffemodel", caffe.TEST)

im = Image.open(image_path)
im = im.resize((227, 227), Image.ANTIALIAS)

# Read image data and transform colormode RGB to BGR
data = np.array(im)[:,:,::-1]

# Subtract training mean
data = data - np.array([104, 117, 123])

# Reshape to Caffe input
# data = np.expand_dims(data, axis=0).transpose(0, 3, 2, 1)
data = np.expand_dims(data, axis=0).transpose(0, 3, 1, 2)

# Output all blobs
out_blobs = net.blobs.keys()
out = net.forward_all(blobs=out_blobs, data=data)

# Store all activations as flat binaries
for b in out_blobs:
    # Reshape to deeplearn.js
    # arr = np.squeeze(out[b], axis=0).transpose(1, 2, 0)
    arr = np.squeeze(out[b], axis=0).transpose(1, 2, 0)
    filename = 'activations/%s' % b
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    f = open(filename, 'wb')
    # Dump to disk as flat binary array
    f.write(arr.astype('float32').tobytes())
    f.close()

# TOP-K
# Output Top-5 values and index
prob = out['prob'].flatten()
indices = np.argsort(-prob.flatten())[:5]
values = prob[indices]
print indices, values
