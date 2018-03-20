import {caffe} from 'caffe-proto';
import {Array1D, Array3D, Array4D, NDArray, NDArrayMath} from 'deeplearn';

// tslint:disable-next-line:max-line-length
export function getLayersFromModel(model: caffe.NetParameter):
    caffe.IV0LayerParameter[]|caffe.IV1LayerParameter[] {
  return model.layer.length > 0 ? model.layer as caffe.IV0LayerParameter[] :
                                  model.layers as caffe.IV1LayerParameter[];
}

function getNumericParam(param: number|number[], defaultValue: number) {
  const p = Array.isArray(param) ? param[0] : param;
  return p || defaultValue;
}

// tslint:disable-next-line:no-any
function isDefined(val: any) {
  return val !== undefined && val !== null;
}

function get1or2dParam(
    param: number|number[], paramW: number, paramH: number,
    defaultValue: number): number|[number, number] {
  if (isDefined(param)) {
    return param as [number, number];
  } else if (isDefined(paramW) && isDefined(paramH)) {
    return [paramW, paramH];
  }
  return defaultValue;
}

/* function getConvStride(param: caffe.ConvolutionParameter):
    number|[number, number] {
  return get1or2dParam(param.stride, param.strideW, param.strideH, 1);
} */

function getPoolStride(param: caffe.PoolingParameter): number|[number, number] {
  return get1or2dParam(param.stride, param.strideW, param.strideH, 1);
}

function getPoolKernel(param: caffe.PoolingParameter): number|[number, number] {
  return get1or2dParam(param.kernelSize, param.kernelW, param.kernelH, 1);
}

function getPoolType(poolType: string|number): number {
  if (typeof poolType === 'number') {
    return poolType;
  } else {
    switch (poolType.toLowerCase()) {
      case 'max':
        return caffe.PoolingParameter.PoolMethod.MAX;
      case 'ave':
        return caffe.PoolingParameter.PoolMethod.AVE;
      default:
        throw TypeError(`Pool type ${poolType} is not implemented`);
    }
  }
}

export function performMathOp(
    math: NDArrayMath, input: NDArray|NDArray[], layer: caffe.ILayerParameter,
    blobs?: NDArray[]): NDArray {
  switch (layer.type.toLowerCase()) {
    case 'input':
    case 'dropout':
      return input as NDArray;

    case 'fc':
    case 'innerproduct':
    case 'inner_product': {
      const innerProductParam =
          caffe.InnerProductParameter.create(layer.innerProductParam);
      const weights = blobs[0] as Array3D;
      const x = (input as Array3D).as1D();
      // const W = weights.as2D(innerProductParam.numOutput, x.shape[0]);
      const W = weights.as2D(weights.shape[1], weights.shape[0]);
      const y = math.matrixTimesVector(W, x);

      if (innerProductParam.biasTerm !== false) {
        const b = blobs[1].as1D() as Array1D;
        return math.add(y, b);
      }
      return y;
    }

    case 'conv':
    case 'convolution': {
      const convolutionParam =
          caffe.ConvolutionParameter.create(layer.convolutionParam);
      const stride = getNumericParam(convolutionParam.stride, 1);

      // TODO throw error if pad is number[] or padW and padH
      // are defined. pad number[] is not supported in dljs
      const pad = getNumericParam(convolutionParam.pad, 0);
      const dimRoundingMode = 'round';

      // kernelSize is estimated from weights implicitly
      const weights = blobs[0] as Array4D;
      const bias = convolutionParam.biasTerm !== false ?
          blobs[1].as1D() as Array1D :
          null;

      return math.conv2d(
          input as Array3D, weights, bias, stride, pad, dimRoundingMode);
    }

    case 'pool':
    case 'pooling': {
      const poolingParam = caffe.PoolingParameter.create(layer.poolingParam);
      const stride = getPoolStride(poolingParam);

      // TODO throw error if pad is number[] or padW and padH
      // are defined. pad number[] is not supported in dljs
      const pad = getNumericParam(poolingParam.pad, 0);
      const dimRoundingMode = 'ceil';

      let kernelSize = getPoolKernel(poolingParam);
      if (poolingParam.globalPooling) {
        kernelSize = (input as Array3D).shape[0];
      }

      switch (getPoolType(poolingParam.pool)) {
        case caffe.PoolingParameter.PoolMethod.MAX:
          return math.maxPool(
              input as Array3D, kernelSize, stride, pad, dimRoundingMode);

        case caffe.PoolingParameter.PoolMethod.AVE:
          return math.avgPool(
              input as Array3D<'float32'>, kernelSize, stride, pad,
              dimRoundingMode);

        default:
          throw TypeError(
              `Pooling type ${poolingParam.pool} is not implemented`);
      }
    }

    case 'batchnorm': {
      const bnParam = caffe.BatchNormParameter.create(layer.batchNormParam);
      const eps = bnParam.eps;
      const mean = blobs[0] as Array3D;
      const variance = blobs[1] as Array3D;

      return math.batchNormalization3D(input as Array3D, mean, variance, eps);
    }

    case 'lrn': {
      const lrnParam = caffe.LRNParameter.create(layer.lrnParam);
      // params need to be converted from caffe to tf.lrn implementation
      const radius = Math.floor(lrnParam.localSize / 2) || 2;
      const bias = lrnParam.k || 1;
      const alpha = lrnParam.alpha / lrnParam.localSize || 1;
      const beta = lrnParam.beta || 0.75;
      const normRegion =
          lrnParam.normRegion === caffe.LRNParameter.NormRegion.WITHIN_CHANNEL ?
          'withinChannel' :
          'acrossChannels';

      return math.localResponseNormalization3D(
          input as Array3D, radius, bias, alpha, beta, normRegion);
    }

    case 'scale': {
      const scaleParam = caffe.ScaleParameter.create(layer.scaleParam);
      const scale = blobs[0] as Array3D;

      let out = math.multiply(input as Array3D, scale);

      if (scaleParam.biasTerm) {
        const bias = blobs[1] as Array3D;
        out = math.add(out as Array3D, bias);
      }

      return out;
    }

    case 'elu':
      return math.elu(input as NDArray);

    case 'relu':
      return math.relu(input as NDArray);

    case 'prelu':
      const alpha = blobs[0].as1D() as Array1D;
      return math.prelu(input as NDArray, alpha);

    case 'tanh':
      return math.tanh(input as NDArray);

    case 'sigmoid':
      return math.sigmoid(input as NDArray);

    case 'softmax':
      const softmaxParam = caffe.SoftmaxParameter.create(layer.softmaxParam);
      const axis = softmaxParam.axis;

      return math.softmax(input as NDArray, axis);

    case 'concat': {
      const inp = input as Array3D[];
      let out = inp[0];
      // Workaround until concat3D(NDArray[]) is supported
      for (let i = 1; i < inp.length; ++i) {
        out = math.concat3D(out, inp[i], 2);
      }
      return out;
    }

    default:
      console.debug(layer);
      throw TypeError(`Layer type ${layer.type} is not implemented`);
  }
}
