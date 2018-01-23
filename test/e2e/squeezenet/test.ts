require('jasmine-co').install();
// make these functions async: afterAll, afterEach, beforeAll, beforeEach, it,
// and fit

import {Array3D, ENV, NDArrayMath, test_util} from 'deeplearn';

import {CaffeModel} from '../../../src';
import * as util from '../../../src/util';

const BASE_PATH = 'base/test/e2e';

// Give the async tests enough time to finish inference
jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000 * 60 * 5;

describe('Squeezenet', () => {
  // force CPU computation
  ENV.setMath(new NDArrayMath('cpu', false));

  const imageUrl = `${BASE_PATH}/assets/cat_227x227.jpg`;

  const activationDir = `${BASE_PATH}/squeezenet/activations/`;
  const modelDir = `${BASE_PATH}/squeezenet/model/`;

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
    'conv1',
    'pool1',
    'fire2/squeeze1x1',
    'fire2/expand1x1',
    'fire2/expand3x3',
    'fire2/concat',
    'fire3/squeeze1x1',
    'fire3/expand1x1',
    'fire3/expand3x3',
    'fire3/concat',
    'pool3',
    'fire4/squeeze1x1',
    'fire4/expand1x1',
    'fire4/expand3x3',
    'fire4/concat',
    'fire5/squeeze1x1',
    'fire5/expand1x1',
    'fire5/expand3x3',
    'fire5/concat',
    'pool5',
    'fire6/squeeze1x1',
    'fire6/expand1x1',
    'fire6/expand3x3',
    'fire6/concat',
    'fire7/squeeze1x1',
    'fire7/expand1x1',
    'fire7/expand3x3',
    'fire7/concat',
    'fire8/squeeze1x1',
    'fire8/expand1x1',
    'fire8/expand3x3',
    'fire8/concat',
    'fire9/squeeze1x1',
    'fire9/expand1x1',
    'fire9/expand3x3',
    'fire9/concat',
    'conv10',
    'pool10',
    'prob'
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
  if (layer.indexOf('conv') !== -1) {
    return `relu_${layer}`;
  } else if (
      layer.indexOf('squeeze') !== -1 || layer.indexOf('expand') !== -1) {
    const p = layer.split('/');
    return `${p[0]}/relu_${p[1]}`;
  }
  return layer;
}
