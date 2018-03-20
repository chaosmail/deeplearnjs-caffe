[![Build Status](https://travis-ci.org/chaosmail/deeplearnjs-caffe.svg?branch=feat%2Ftest)](https://travis-ci.org/chaosmail/deeplearnjs-caffe)

# Deeplearn-Caffe

Run pretrained Caffe models in the browser with GPU support via the wonderful [deeplearn.js][deeplearn] library. This package provides utility tools and a model loader for Caffe models to support the following tasks:

* Loading and parsing `*.caffemodel` files into deeplearn.js weights
* Loading and parsing `*.binaryproto` files into deeplearn.js blobs
* Loading and parsing `*.prototxt` files into deeplearn.js models

## Usage

### Installation
You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/deeplearn-caffe"></script>
```

Then loading model is a simple as referencing the path to the caffemodel and prototxt files.

Here is an example of loading GoogLeNet:

```js
var GITHUB_CDN = 'https://rawgit.com/';
var MODEL_DIR = 'models/';

// Caffemodel needs to be downloaded from here
var modelUrl = 'http://dl.caffe.berkeleyvision.org';

var prototxtUrl = GITHUB_CDN + 'BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt';
var caffemodelUrl = MODEL_DIR + 'bvlc_googlenet.caffemodel';

// Initialize the CaffeModel
var model = new deeplearnCaffe.CaffeModel(caffemodelUrl, prototxtUrl);
```

This is how you load Squeezenet directly from Github:

```js
// The model is served entirely from Github
var GITHUB_CDN = 'https://rawgit.com/';

var prototxtUrl = GITHUB_CDN + 'DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt';
var caffemodelUrl = GITHUB_CDN + 'DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel';

// Initialize the CaffeModel
var model = new deeplearnCaffe.CaffeModel(caffemodelUrl, prototxtUrl);
```

### Run Demos

To run the demo, use the following:

```bash
npm run build

# Start a webserver
npm run start
```

Now navigate to http://localhost:8080/demos.

> Hint: some of the models are quite big (>30MB). You have to download the caffemodel files and place them into the `demos/models` directory to save bandwith.

## Development

```sh
npm install
```

To build a standalone bundle run

```sh
npm run build
```

[deeplearn]: https://github.com/PAIR-code/deeplearnjs
