import {caffe} from 'caffe-proto';
import * as dag from 'dag-iterator';
import {Array1D, Array3D, ENV, Model, NDArray} from 'deeplearn';
// tslint:disable-next-line:max-line-length
import {convBlobToNDArray, getAllVariables, getPreprocessDim, getPreprocessOffset} from './blob';
import {layersToDagEdges, layersToDagNodes} from './dag';
import {performMathOp} from './layer';
import * as util from './util';

export class CaffeModel implements Model {
  /**
   * Model weights per layer
   */
  protected variables: {[varName: string]: NDArray[]};

  /**
   * Preprocessing Offset
   */
  protected preprocessOffset: Array1D|Array3D;

  /**
   * Preprocessing Dimensions
   */
  protected preprocessDim: number;

  /**
   * Model DAG Nodes
   */
  private nodes: Array<dag.INode<caffe.ILayerParameter>>;

  /**
   * Model DAG Edges
   */
  private edges: dag.IEdge[];

  /**
   * Constructor
   * @param caffemodelUrl url to the caffemodel file
   * @param prototxtUrl url to the prototxt file
   */
  constructor(
      private caffemodelUrl: string, private prototxtUrl: string,
      private meanBinaryprotoUrl?: string) {}

  /**
   * Set the preprocessing offset
   * @param offset training mean
   */
  setPreprocessOffset(offset: Array1D|Array3D) {
    this.preprocessOffset = offset;
  }

  /**
   * Get the preprocessing offset
   */
  getPreprocessOffset(): Array1D|Array3D {
    return this.preprocessOffset;
  }

  /**
   * Set the preprocessing dimensions
   * @param dim height/width of the input dimension
   */
  setPreprocessDim(dim: number) {
    this.preprocessDim = dim;
  }

  /**
   * Get the preprocessing dimensions
   */
  getPreprocessDim(): number {
    return this.preprocessDim;
  }

  /**
   * Set the nodes of the model graph
   * @param nodes array of nodes
   */
  setNodes(nodes: Array<dag.INode<caffe.ILayerParameter>>) {
    this.nodes = nodes;
  }

  /**
   * Get the nodes of the model graph
   */
  getNodes(): Array<dag.INode<caffe.ILayerParameter>> {
    return this.nodes;
  }

  /**
   * Set the edges of the model graph
   * @param edges array of edges
   */
  setEdges(edges: dag.IEdge[]) {
    this.edges = edges;
  }

  /**
   * Get the edges of the model graph
   */
  getEdges(): dag.IEdge[] {
    return this.edges;
  }

  /**
   * Load the model
   */
  load() {
    const tasks = [];

    if (this.caffemodelUrl) {
      tasks.push(this.load_caffemodel(this.caffemodelUrl));
    }
    if (this.prototxtUrl) {
      tasks.push(this.load_prototxt(this.prototxtUrl));
    }
    if (this.meanBinaryprotoUrl) {
      tasks.push(this.load_binaryproto(this.meanBinaryprotoUrl));
    }
    return Promise.all(tasks);
  }

  /**
   * Load the .caffemodel file and parse the weights it into variables
   */
  load_caffemodel(uri: string) {
    return util.fetchArrayBuffer(uri)
        .then(util.parseCaffemodel)
        .then((model) => {
          this.variables = getAllVariables(model);

          // read the cropping dimensions
          const dim = getPreprocessDim(model);
          if (dim) {
            this.setPreprocessDim(dim);
          }

          // read the training mean
          const offset = getPreprocessOffset(model);
          if (offset) {
            this.setPreprocessOffset(offset as Array1D);
          }
        });
  }

  /**
   * Load the .prototxt file and parse it into DAG nodes and edges
   */
  load_prototxt(uri: string) {
    return util.fetchText(uri).then(util.parsePrototxt).then((model) => {
      this.setEdges(layersToDagEdges(model));
      this.setNodes(layersToDagNodes(model));
    });
  }

  /**
   * Overwrite in child class to load additional resources
   */
  load_binaryproto(uri: string) {
    return util.fetchArrayBuffer(uri)
        .then(util.parseBlob)
        .then((trainingMean) => {
          const offset: NDArray = convBlobToNDArray(trainingMean);
          this.setPreprocessOffset(
              offset.as3D(offset.shape[0], offset.shape[1], offset.shape[2]));
        });
  }

  predict(
      input: NDArray, untilLayer?: string,
      cb?: (name: string, layer: caffe.ILayerParameter, activation: NDArray) =>
          void): NDArray {
    const math = ENV.math;

    // Keep a map of named activations for rendering purposes.
    const namedActivations: {[key: string]: NDArray} = {};
    let currAct: NDArray|NDArray[] = input;

    dag.iterateDfs<caffe.ILayerParameter>(
        this.nodes, this.edges,
        (layer: caffe.ILayerParameter, parents: caffe.ILayerParameter[],
         i: number, depth: number) => {
          if (i === 0) {
            // Convert input to Array3D<float32>
            currAct = (currAct as Array3D).asType('float32');

            // Convert RGB to BGR color mode
            currAct = math.reverse3D(currAct as Array3D, -1);

            // Resize the input image
            if (this.preprocessDim) {
              currAct = math.resizeBilinear3D(
                  currAct as Array3D, [this.preprocessDim, this.preprocessDim]);
            }

            // Subtract training mean from input image
            if (this.preprocessOffset &&
                this.preprocessOffset.dataSync().length > 0) {
              currAct =
                  math.subtract(currAct as Array3D, this.preprocessOffset) as
                  Array3D;
            }
          } else if (parents.length === 1) {
            currAct = namedActivations[parents[0].name];
          } else if (parents.length > 1) {
            currAct = parents.map((d) => namedActivations[d.name]);
          }

          currAct = performMathOp(
              math, currAct, layer, this.variables[`${layer.name}`]);

          namedActivations[layer.name] = currAct as NDArray;

          if (cb) {
            cb(layer.name, layer, currAct);
          }
        },
        untilLayer);

    return currAct;
  }

  dispose() {
    this.preprocessOffset.dispose();
    for (const varName in this.variables) {
      this.variables[varName].map((d) => d.dispose());
    }
  }
}
