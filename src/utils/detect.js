import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";
import labels from "./labels.json";
// import React, { useState } from "react";

const numClass = labels.length;
let numberOfPersons = 0;
let personsData = {};

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

  return [input, xRatio, yRatio];
};

/**
 * Function run inference and do detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback function to run after detection process
 */
export const detect = async (source, model, canvasRef, callback = () => { }) => {

  let numberOfPersons = 0;

  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight); // preprocess image

  const res = model.net.execute(input); // inference model
  const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
    const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2
          tf.add(x1, w), //x2
        ],
        2
      )
      .squeeze();
  }); // process boxes [y1, x1, y2, x2]

  const [scores, classes] = tf.tidy(() => {
    // class scores
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
    return [rawScores.max(1), rawScores.argMax(1)];
  }); // get max scores and classes index

  const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes

  numberOfPersons = nms.size; // Update jumlah orang
  const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
  const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
  const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

  const currentTime = new Date().getTime();

  Object.keys(personsData).forEach((personKey) => {
    personsData[personKey].present = false;
  });

  for (let i = 0; i < numberOfPersons; i++) {
    const personId = nms.dataSync()[i]; // Use dataSync to get values from the tensor
    const personKey = `person_${personId}`;

    if (!personsData[personKey]) {
      // Initialize person's data if not present
      personsData[personKey] = {
        firstDetectionTime: currentTime,
        lastDetectionTime: currentTime,
        totalDuration: 0,
        present: true,
      };
    } else {
      // Update last detection time for the person
      personsData[personKey].lastDetectionTime = currentTime;

      // Mark the person as present in the current frame
      personsData[personKey].present = true;
    }
  }

  Object.keys(personsData).forEach((personKey) => {
    if (!personsData[personKey].present) {
      const duration = currentTime - personsData[personKey].lastDetectionTime;
      personsData[personKey].totalDuration += duration;
    }
  });


  renderBoxes(canvasRef, boxes_data, scores_data, classes_data, [xRatio, yRatio]); // render boxes



  tf.dispose([res, transRes, boxes, scores, classes, nms,]); // clear memory

  // callback();
  callback(numberOfPersons);

  tf.engine().endScope(); // end of scoping

  return numberOfPersons;
};

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectVideo = (vidSource, model, canvasRef, setNumberOfPersons) => {
  /**
   * Function to detect every frame from video
   */
  const detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      return;
    }

    try {
      const detectedPersons = await detect(vidSource, model, canvasRef);
      console.log("Number of persons detected:", detectedPersons);

      // Iterate through each person's data and log their total duration
      Object.keys(personsData).forEach((key) => {
        const personData = personsData[key];
        console.log(`Person ${key} total duration: ${personData.totalDuration} milliseconds`);
      });

      // setNumberOfPersons(detectedPersons);
    } catch (error) {
      console.error("Error during detection:", error);
    }


    requestAnimationFrame(detectFrame);

    // const detectedPersons = await detect(vidSource, model, canvasRef);
    // console.log("Number of persons detected:", detectedPersons);
    // setNumberOfPersons(detectedPersons);

    // get another frame
  };

  detectFrame(); // initialize to detect every frame
};
