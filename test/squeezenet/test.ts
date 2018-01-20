require('jasmine-co').install();

import {test_util} from 'deeplearn';

import {CaffeModel} from '../../src';
import * as util from '../../src/util';
import {Array3D} from 'deeplearn/dist/math/ndarray';

describe('Squeezenet', () => {
  const ACTIVATION_DIR = 'base/test/squeezenet/activations/';
  const MODEL_DIR = 'base/test/squeezenet/model/';

  const prototxtUrl = MODEL_DIR + 'net.prototxt';
  const caffemodelUrl = MODEL_DIR + 'net.caffemodel';

  // Initialize the CaffeModel
  const model = new CaffeModel(caffemodelUrl, prototxtUrl);

  beforeAll(async () => {
    // Load the model weights
    await model.load();
  });

  const layers = [
    'conv1',
    'relu_conv1',
    'pool1',
    'fire2/squeeze1x1',
    'fire2/relu_squeeze1x1',
    'fire2/expand1x1',
    'fire2/relu_expand1x1',
    'fire2/expand3x3',
    'fire2/relu_expand3x3',
    'fire2/concat',
    'fire3/squeeze1x1',
    'fire3/relu_squeeze1x1',
    'fire3/expand1x1',
    'fire3/relu_expand1x1',
    'fire3/expand3x3',
    'fire3/relu_expand3x3',
    'fire3/concat',
    'pool3',
    'fire4/squeeze1x1',
    'fire4/relu_squeeze1x1',
    'fire4/expand1x1',
    'fire4/relu_expand1x1',
    'fire4/expand3x3',
    'fire4/relu_expand3x3',
    'fire4/concat',
    'fire5/squeeze1x1',
    'fire5/relu_squeeze1x1',
    'fire5/expand1x1',
    'fire5/relu_expand1x1',
    'fire5/expand3x3',
    'fire5/relu_expand3x3',
    'fire5/concat',
    'pool5',
    'fire6/squeeze1x1',
    'fire6/relu_squeeze1x1',
    'fire6/expand1x1',
    'fire6/relu_expand1x1',
    'fire6/expand3x3',
    'fire6/relu_expand3x3',
    'fire6/concat',
    'fire7/squeeze1x1',
    'fire7/relu_squeeze1x1',
    'fire7/expand1x1',
    'fire7/relu_expand1x1',
    'fire7/expand3x3',
    'fire7/relu_expand3x3',
    'fire7/concat',
    'fire8/squeeze1x1',
    'fire8/relu_squeeze1x1',
    'fire8/expand1x1',
    'fire8/relu_expand1x1',
    'fire8/expand3x3',
    'fire8/relu_expand3x3',
    'fire8/concat',
    'fire9/squeeze1x1',
    'fire9/relu_squeeze1x1',
    'fire9/expand1x1',
    'fire9/relu_expand1x1',
    'fire9/expand3x3',
    'fire9/relu_expand3x3',
    'fire9/concat',
    'drop9',
    'conv10',
    'relu_conv10',
    'pool10',
    'prob'
  ];

  for (let i = 0; i <= layers.length; ++i) {
    const layer = layers[i];

    it(layer, async () => {
      const buffer = await util.fetchArrayBuffer(ACTIVATION_DIR + layer);
      const expected = new Float32Array(buffer);

      const x: Array3D = null;
      const output = await model.predict(x, layer);
      const actual = output.dataSync();

      test_util.expectArraysClose(actual, expected);
    });
  }
});
