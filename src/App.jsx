import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";



const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape
  const [dateState, setDateState] = useState(new Date());

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const [numberOfPersons, setNumberOfPersons] = useState(0);
  const [logs, setLogs] = useState([]);

  // model configs
  const modelName = "yolov8n";

  useEffect(() => {
    setDateState(new Date(), 10000);
    tf.ready().then(async () => {
      const yolov8 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions });
          },
        }
      );

      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      const warmupResults = yolov8.execute(dummyInput);

      // console.log = (message) => {
      //   // Menambahkan pesan ke state logs
      //   setLogs((prevLogs) => [...prevLogs, message]);
  
      //   // Memproses pesan untuk mendapatkan jumlah orang yang terdeteksi
      //   if (message.includes('Number of persons detected:')) {
      //     // Menggunakan ekstraksi angka dengan regex untuk menghindari NaN
      //     const countMatch = message.match(/(\d+)/);
      //     const count = countMatch ? parseInt(countMatch[0], 10) : 0;
      //     // setNumberOfPersons(count);
      //   }
  
      //   // Memanggil fungsi console.log asli
      //   originalConsoleLog(message);
      // };

      

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
      });

      // detectVideo(videoRef.current, yolov8, canvasRef.current, setNumberOfPersons);
      tf.dispose([warmupResults, dummyInput]);
    });
  }, [numberOfPersons]);


  return (
    <div className="App" class="">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="p-6">
        <div className="header">
          <h1 className="font-semibold my-10">Chill Detection Aplication</h1>
          <p>Date Time :
          {' '}
          {dateState.toLocaleDateString('en-GB', {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
          })}
          {' '}
          {dateState.toLocaleString('en-US', {
            hour: 'numeric',
            minute: 'numeric',
            hour12: true,
          })}
        </p>
          {/* <p>
            Model Name : <code className="code">{modelName}</code>
          </p> */}
        </div>

        <div className="content">
          <img
            src="#"
            ref={imageRef}
            onLoad={() => detect(imageRef.current, model, canvasRef.current)}
          />
          <video
            autoPlay
            muted
            ref={cameraRef}
            onPlay={() => detectVideo(cameraRef.current, model, canvasRef.current)}
          />
          <video
            autoPlay
            muted
            ref={videoRef}
            onPlay={() => detectVideo(videoRef.current, model, canvasRef.current)}
          />
          <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef} />
        </div>

        <div className="my-8 flex justify-center">
          <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
        </div>
      </div>


      

    </div>

  );

};
export default App;
