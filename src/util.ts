import {caffe} from 'caffe-proto';
import {Array1D, NDArrayMath} from 'deeplearn';
import * as prototxtParser from 'prototxt-parser';

// tslint:disable-next-line:no-any
export function isNotNull(val: any): boolean {
  return val !== undefined && val !== null;
}

export function fetchText(uri: string): Promise<string> {
  return fetch(new Request(uri))
      .then(handleFetchErrors)
      .then((res) => res.text());
}

export function fetchArrayBuffer(uri: string): Promise<ArrayBuffer> {
  return fetch(new Request(uri))
      .then(handleFetchErrors)
      .then((res) => res.arrayBuffer());
}

function handleFetchErrors(response: Response) {
  if (!response.ok) {
    throw Error(response.statusText);
  }
  return response;
}

export function parseBlob(data: ArrayBuffer) {
  return caffe.BlobProto.decode(new Uint8Array(data));
}

export function parseCaffemodel(data: ArrayBuffer) {
  return caffe.NetParameter.decode(new Uint8Array(data));
}

export function parsePrototxt(data: string) {
  const params = snakeToCamel(prototxtParser.parse(data));
  return caffe.NetParameter.create(params);
}

// camelize string
const camelize = (str: string) => {
  const c = (m: string, i: number) =>
      i === 0 ? m : m.charAt(0).toUpperCase() + m.slice(1);
  return str.split('_').map(c).join('');
};

// convert object with snake case properties to camel case
// tslint:disable-next-line:no-any
function snakeToCamel(obj: any): any {
  // Check if obj is an array
  if (Array.isArray(obj)) {
    return obj.map(snakeToCamel);
  }
  // check if obj is an object
  else if (obj === Object(obj)) {
    for (const key in obj) {
      // skip loop if the property is from prototype
      if (!obj.hasOwnProperty(key)) continue;
      const newKey = camelize(key);
      obj[newKey] = snakeToCamel(obj[key]);
      if (newKey !== key) {
        delete obj[key];
      }
    }
  }
  return obj;
}

export function loadImageData(
    url: string, elem?: HTMLImageElement): Promise<HTMLImageElement> {
  const img = elem || document.createElement('img') as HTMLImageElement;

  return new Promise((resolve, reject) => {
    img.onload = async () => {
      resolve(img);
    };
    img.onerror = (err) => {
      reject(err);
    };
    img.src = url;
  });
}

export interface ITopK {
  indices: Int32Array;
  values: Float32Array;
}

/**
 * Get the topK classes for pre-softmax logits. Returns a map of className
 * to softmax normalized probability.
 *
 * @param logits Pre-softmax logits array.
 * @param topK How many top classes to return.
 */
export async function getTopK(prob: Array1D, topK: number): Promise<ITopK> {
  const values = await prob.data();
  const input = Array1D.new(new Float32Array(values));
  const math = new NDArrayMath('cpu', false);
  const topk = math.topK(input, topK);
  const topkIndices = await topk.indices.data() as Int32Array;
  const topkValues = await topk.values.data() as Float32Array;

  return {indices: topkIndices, values: topkValues};
}
