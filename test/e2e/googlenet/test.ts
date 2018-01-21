require('jasmine-co').install();
// make these functions async: afterAll, afterEach, beforeAll, beforeEach, it,
// and fit

import {Array3D, ENV, NDArrayMath, test_util} from 'deeplearn';

import {CaffeModel} from '../../../src';
import * as util from '../../../src/util';

const BASE_PATH = 'base/test/e2e';

// Give the async tests enough time to finish inference
jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000 * 60 * 5;

describe('GoogLeNet', () => {
  // force CPU computation
  ENV.setMath(new NDArrayMath('cpu', false));

  const imageUrl = `${BASE_PATH}/assets/cat.jpg`;

  const activationDir = `${BASE_PATH}/googlenet/activations/`;
  const modelDir = `${BASE_PATH}/googlenet/model/`;

  const prototxtUrl = modelDir + 'net.prototxt';
  const caffemodelUrl = modelDir + 'net.caffemodel';

  // Initialize the CaffeModel
  const model = new CaffeModel(caffemodelUrl, prototxtUrl);

  beforeAll(async () => {
    // Load the model weights
    await model.load();
  });


  // All layers that produce an output blob in caffe.
  // Hint: Self loop layers (like ReLu, Dropout, etc.) produce an
  // output blob e.g. 'conv1' which equals to Dljs output of 'relu_conv1'
  const layers = [
    'conv1/7x7_s2',
    'pool1/3x3_s2',
    'pool1/norm1',
    'conv2/3x3_reduce',
    'conv2/3x3',
    'conv2/norm2',
    'pool2/3x3_s2',
    'inception_3a/1x1',
    'inception_3a/3x3_reduce',
    'inception_3a/3x3',
    'inception_3a/5x5_reduce',
    'inception_3a/5x5',
    'inception_3a/pool',
    'inception_3a/pool_proj',
    'inception_3a/output',
    'inception_3b/1x1',
    'inception_3b/3x3_reduce',
    'inception_3b/3x3',
    'inception_3b/5x5_reduce',
    'inception_3b/5x5',
    'inception_3b/pool',
    'inception_3b/pool_proj',
    'inception_3b/output',
    'pool3/3x3_s2',
    'inception_4a/1x1',
    'inception_4a/3x3_reduce',
    'inception_4a/3x3',
    'inception_4a/5x5_reduce',
    'inception_4a/5x5',
    'inception_4a/pool',
    'inception_4a/pool_proj',
    'inception_4a/output',
    'inception_4b/1x1',
    'inception_4b/3x3_reduce',
    'inception_4b/3x3',
    'inception_4b/5x5_reduce',
    'inception_4b/5x5',
    'inception_4b/pool',
    'inception_4b/pool_proj',
    'inception_4b/output',
    'inception_4c/1x1',
    'inception_4c/3x3_reduce',
    'inception_4c/3x3',
    'inception_4c/5x5_reduce',
    'inception_4c/5x5',
    'inception_4c/pool',
    'inception_4c/pool_proj',
    'inception_4c/output',
    'inception_4d/1x1',
    'inception_4d/3x3_reduce',
    'inception_4d/3x3',
    'inception_4d/5x5_reduce',
    'inception_4d/5x5',
    'inception_4d/pool',
    'inception_4d/pool_proj',
    'inception_4d/output',
    'inception_4e/1x1',
    'inception_4e/3x3_reduce',
    'inception_4e/3x3',
    'inception_4e/5x5_reduce',
    'inception_4e/5x5',
    'inception_4e/pool',
    'inception_4e/pool_proj',
    'inception_4e/output',
    'pool4/3x3_s2',
    'inception_5a/1x1',
    'inception_5a/3x3_reduce',
    'inception_5a/3x3',
    'inception_5a/5x5_reduce',
    'inception_5a/5x5',
    'inception_5a/pool',
    'inception_5a/pool_proj',
    'inception_5a/output',
    'inception_5b/1x1',
    'inception_5b/3x3_reduce',
    'inception_5b/3x3',
    'inception_5b/5x5_reduce',
    'inception_5b/5x5',
    'inception_5b/pool',
    'inception_5b/pool_proj',
    'inception_5b/output',
    'pool5/7x7_s1',
    'loss3/classifier',
    'prob',
  ];

  layers.forEach((layer) => {
    it(layer, async () => {
      // Load the image data
      const imageData = await util.loadImageData(imageUrl);
      const input = Array3D.fromPixels(imageData);

      // Make a forward pass through model until the specified layer
      // TODO this could be done much faster, by hooking into the callback
      // function of model.predict()
      const output = await model.predict(input, getDljsOutputLayer(layer));
      const actual = output.dataSync();

      // Load the results from caffe
      const buffer = await util.fetchArrayBuffer(activationDir + layer);
      const expected = new Float32Array(buffer);

      expect(actual).toBeDefined();
      expect(actual).not.toBeNull();
      expect(expected).toBeDefined();
      expect(expected).not.toBeNull();

      test_util.expectArraysClose(actual, expected);
    });
  });
});

function getDljsOutputLayer(layer: string) {
  if (layer.indexOf('pool') !== -1 && layer.indexOf('proj') !== -1) {
    return layer;
  } else if (
      layer.indexOf('conv') !== -1 || layer.indexOf('proj') !== -1 ||
      layer.indexOf('1x1') !== -1 || layer.indexOf('3x3') !== -1 ||
      layer.indexOf('5x5') !== -1) {
    const p = layer.split('/');
    return `${p[0]}/relu_${p[1]}`;
  }
  return layer;
}
