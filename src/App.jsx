import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detectImage, detectVideo } from "./utils/detect";
import { cropImage } from './utils/cropImage';
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape
  const [boundingBoxes, setBoundingBoxes] = useState([])

  const [classificator, setClassificator] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // model configs
  const modelName = "yolov5n";
  const classThreshold = 0.2;

  useEffect(() => { // Carregar yolo
    tf.ready().then(async () => {
      const yolov5 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      const dummyInput = tf.ones(yolov5.inputs[0].shape);
      const warmupResult = await yolov5.executeAsync(dummyInput);
      tf.dispose(warmupResult); // cleanup memory
      tf.dispose(dummyInput); // cleanup memory

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov5,
        inputShape: yolov5.inputs[0].shape,
      }); // set model & input shape
    });
  }, []);

  useEffect(() => { // Carregar classificador
    tf.ready().then(async () => {
      const mobileNet = await tf.loadLayersModel(
        `${window.location.href}/classificator/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model

      const inputShape = mobileNet.inputs[0].shape.map(dim => dim || 1);
      const dummyInput = tf.ones(inputShape);
      const warmupResult = await mobileNet.predict(dummyInput);
      // console.log(warmupResult)
      tf.dispose(warmupResult); // cleanup memory
      tf.dispose(dummyInput); // cleanup memory

      setLoading({ loading: false, progress: 1 });
      setClassificator({
        net: mobileNet,
        inputShape: mobileNet.inputs[0].shape,
      }); // set model & input shape
    });
  }, []);

  useEffect(() => {
    console.log(imageRef.current)
  }, [imageRef])

  const handleDetectImage = async () => {
    // Obtenha a razão entre as dimensões da imagem original e um tamanho máximo desejado
    let maxWidth = 720; // Defina isso para o max-width desejado
    let maxHeight = 500; // Defina isso para o max-height desejado
    const ratioWidth = imageRef.current.width / maxWidth;
    const ratioHeight = imageRef.current.height / maxHeight;
    const ratio = Math.max(ratioWidth, ratioHeight);

    // Calcule as novas dimensões da imagem
    const newWidth = imageRef.current.width / ratio;
    const newHeight = imageRef.current.height / ratio;

    // Crie um novo elemento canvas para redimensionar a imagem
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = newWidth;
    resizeCanvas.height = newHeight;
    const resizeCtx = resizeCanvas.getContext('2d');

    // Desenhe a imagem original no canvas de redimensionamento
    resizeCtx.drawImage(imageRef.current, 0, 0, newWidth, newHeight);

    // Agora use o canvas de redimensionamento em vez da imagem original
    const source = resizeCanvas;

    const bb = await detectImage(source, model, classThreshold, canvasRef.current);
    console.log(bb); // Imprima a variável 'bb'
    setBoundingBoxes(bb);

    // Crie um novo elemento canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Defina a largura e a altura do canvas para corresponder à caixa delimitadora
    maxWidth = Math.min(bb[0].width, 720); // limite a largura máxima
    maxHeight = Math.min(bb[0].height, 500); // limite a altura máxima
    canvas.width = maxWidth;
    canvas.height = maxHeight;

    // Desenhe a parte do canvas original que corresponde à caixa delimitadora no novo canvas
    ctx.drawImage(
      source, // o canvas original
      bb[0].x1, bb[0].y1, // as coordenadas de início do corte no canvas original
      maxWidth, maxHeight, // a largura e a altura do corte
      0, 0, // as coordenadas de início do desenho no novo canvas
      maxWidth, maxHeight // a largura e a altura do desenho
    );

    // Crie uma URL de objeto a partir do canvas
    const url = canvas.toDataURL();

    // Crie um novo elemento de imagem
    const img = document.createElement('img');

    // Defina o src do elemento de imagem para a URL do objeto
    img.src = url;

    // Adicione o novo elemento de imagem ao documento
    document.body.appendChild(img);
  };



  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="header">
        <h1>📷 YOLOv5 Live Detection App</h1>
        <p>
          YOLOv5 live detection application on browser powered by <code>tensorflow.js</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => handleDetectImage()}
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() => detectVideo(cameraRef.current, model, classThreshold, canvasRef.current)}
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => detectVideo(videoRef.current, model, classThreshold, canvasRef.current)}
        />
        <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef} />
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;
