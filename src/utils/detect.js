import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height
    console.log("w,h => preprocess", w, h)
    const maxSize = Math.max(w, h); // get max size
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
      .div(255.0) // normalize
      .expandDims(0); // add batch
  });
  console.log("preprocess", input, xRatio, yRatio)
  return [input, xRatio, yRatio];
};

/**
 * Function to detect image.
 * @param {HTMLImageElement} imgSource image source
 * @param {tf.GraphModel} model loaded YOLOv5 tensorflow.js model
 * @param {Number} classThreshold class threshold
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @returns {Promise<Array>}
 */
export const detectImage = async (imgSource, model, classThreshold, canvasRef) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height
  let boundingBox = []
  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(imgSource, modelWidth, modelHeight);

  await model.net.executeAsync(input).then((res) => {
    const [boxes, scores, classes] = res.slice(0, 3);
    const boxes_data = boxes.dataSync();
    const scores_data = scores.dataSync();
    const classes_data = classes.dataSync();
    boundingBox = renderBoxes(canvasRef, imgSource, classThreshold, boxes_data, scores_data, classes_data, [xRatio, yRatio]); // render boxes
    tf.dispose(res); // clear memory

  });
  tf.engine().endScope(); // end of scoping
  return boundingBox
};

export const classifyImage = async (imgSource, model) => {
  // Obtenha as dimensões do modelo
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);

  // Inicie um escopo de TensorFlow para gerenciamento de memória
  tf.engine().startScope();

  try {
    // Pré-processamento da imagem
    const [input, _, __] = preprocess(imgSource, modelWidth, modelHeight);

    // Execute a previsão
    const predictions = await model.net.predict(input);

    // Obtenha os resultados como array
    const predictionArray = predictions.dataSync();

    // Libere a memória usada por `input` e `predictions`
    tf.dispose([input, predictions]);

    // Encerre o escopo do TensorFlow
    tf.engine().endScope();

    // Retorne os resultados da previsão
    return predictionArray;
  } catch (error) {
    console.error("Erro ao classificar imagem:", error);
    tf.engine().endScope(); // Encerre o escopo em caso de erro
    return null; // ou outra forma de lidar com o erro
  }
};


/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv5 tensorflow.js model
 * @param {Number} classThreshold class threshold
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectVideo = (vidSource, model, classThreshold, canvasRef) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  /**
   * Function to detect every frame from video
   */
  const detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
      return; // handle if source is closed
    }

    tf.engine().startScope(); // start scoping tf engine
    const [input, xRatio, yRatio] = preprocess(vidSource, modelWidth, modelHeight);

    await model.net.executeAsync(input).then((res) => {
      const [boxes, scores, classes] = res.slice(0, 3);
      const boxes_data = boxes.dataSync();
      const scores_data = scores.dataSync();
      const classes_data = classes.dataSync();
      renderBoxes(canvasRef, classThreshold, boxes_data, scores_data, classes_data, [
        xRatio,
        yRatio,
      ]); // render boxes
      tf.dispose(res); // clear memory
    });

    requestAnimationFrame(detectFrame); // get another frame
    tf.engine().endScope(); // end of scoping
  };

  detectFrame(); // initialize to detect every frame
};
