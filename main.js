var inputTensor;
var inputTest = [];
var model;
var max = Number.MIN_VALUE;
var min = Number.MAX_VALUE;

var inputMax = 0;
var inputMin = 0;

function int32_to_float32(value) {
  const binaryArray = new Int8Array([value[0], value[1], value[2], value[3]]);

  // Create a DataView to access the binary data
  const dataView = new DataView(binaryArray.buffer);

  // Read the Float32 value at byte offset 0
  return dataView.getFloat32(0);
}

function float32ToInt8(floatValue) {
  let intArray = new Int8Array(4);

  for (var i = 0; i < 4; i++) {
    intArray[i] = floatValue >> 2;
  }

  return intArray;
}

function arrayFloat32_to_ArrayInt(array) {
  let reuslt = new Array(array.length * 4).fill(0);
  let tmp;
  let index = 0;
  for (let i = 0; i < array.length; i++) {
    tmp = float32ToInt8(array[i]);
    reuslt[index] = tmp[0];
    reuslt[index + 1] = tmp[1];
    reuslt[index + 2] = tmp[2];
    reuslt[index + 3] = tmp[3];
    index = index + 4;
  }

  return reuslt;
}

async function read_file_f32(file) {
  const buffer = await file.arrayBuffer();
  let dataview = new DataView(buffer);
  let reuslt = new Float32Array(Math.ceil(file.size / 4));

  let num1 = 0;
  let num2 = 0;
  let num3 = 0;
  let num4 = 0;

  let index = 0;
  let tmp = 0;
  max = Number.MIN_VALUE;
  min = Number.MAX_VALUE;

  for (var i = 0; i < file.size; i = i + 4) {
    num1 = dataview.getInt8(i);
    if (i + 1 < file.size) {
      num2 = dataview.getInt8(i + 1);
    } else {
      num2 = 0;
      num3 = 0;
      num4 = 0;
    }

    if (i + 2 < file.size) {
      num3 = dataview.getInt8(i + 2);
    } else {
      num3 = 0;
      num4 = 0;
    }

    if (i + 3 < file.size) {
      num4 = dataview.getInt8(i + 3);
    } else {
      num4 = 0;
    }

    //tmp = (num1 << 24) | (num2 << 16) | (num3 << 8) | num4; // int8 to int32

    //console.log(tmp, index);
    tmp = int32_to_float32([num1, num2, num3, num4]);
    if (tmp >= max) {
      max = tmp;
    }
    if (tmp <= min) {
      min = tmp;
    }

    reuslt[index] = int32_to_float32([num1, num2, num3, num4]);
    index++;
  }
  return reuslt;
}

async function read_file_int16(file) {
  const buffer = await file.arrayBuffer();
  let buffer16 = new Int16Array(buffer);
  let reuslt = new Float32Array(buffer.byteLength / 2);

  for (let i = 0; i < buffer16.length; i++) {
    reuslt[i] = buffer16[i];
  }
  return reuslt;
}

function binaryCrossentropyLoss(yTrue, yPred) {
  // Compute binary cross-entropy loss between yTrue and yPred
  const loss = tf.losses.logLoss(yTrue, yPred);

  // Return the loss as a scalar value
  return loss.mean();
}

function customLoss(yTrue, yPred) {
  const input = yPred.arraySync();
  const yPred8 = arrayFloat32_to_ArrayInt(yPred.arraySync()[0]);
  console.log("yPred8", yPred8);
  // Clipping to prevent NaN and Inf values
  //yPred = tf.clipByValue(yPred, 1e-7, 1 - 1e-7);
  //yPred = tf.clipByValue(yPred, min, max);

  // Compute binary cross-entropy loss
  const bceLoss = tf.metrics.binaryCrossentropy(yTrue, yPred);

  // Compute L1L2 regularization penalty on model weights
  const regularization = tf.regularizers.l1l2({ l1: 0.01, l2: 0.01 });
  const regLoss = regularization.apply(model.trainableWeights);

  // Compute total loss as sum of binary cross-entropy loss and weight penalty
  const totalLoss = bceLoss.add(regLoss);

  return totalLoss;
}

async function inputFile(event) {
  event.preventDefault();

  console.log(event.target[0].files[0]);

  //inputTensor = await read_file_f32(event.target[0].files[0]);
  inputTensor = await read_file_int16(event.target[0].files[0]);

  for (let index = 0; index < inputTensor.length && index < 10; index++) {
    inputTest.push(inputTensor[index]);
  }

  console.log("MAX", max, "MIN", min);
  const size = inputTensor.length;
  console.log("read_file_int16", inputTensor, size);

  const input = tf.input({
    shape: [size],
    /// dtype: "string"
  });

  const units = Math.round(size / 64);
  const compress = tf.layers
    .dense({
      units: 1,//units <= 0 ? 1 : units,
      //activation: "relu",
      //dtype: "float32",
      //useBias: true,
    })
    .apply(input);

  const out = tf.layers
    .dense({
      units: size,
      //activation: "linear",
      //dtype: "int32",
      //useBias: true,
    })
    .apply(compress);

  model = tf.model({ inputs: input, outputs: out });
  model.summary();
  tfvis.show.modelSummary(surface, model);

  inputTensor = tf.tensor1d(inputTensor);
  inputTensor = inputTensor.reshape([1, size]);
  console.log("inputTensor", inputTensor);

  inputMax = inputTensor.max();
  inputMin = inputTensor.min();
  console.log("inputMax");
  inputMax.print();
  console.log("inputMin");
  inputMin.print();

  /*const fileNIntArray = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

  console.log(fileNIntArray.print());
  console.log(fileNIntArray.max().print(), fileNIntArray.min().print());

  inputTensor = fileNIntArray.reshape([1, size]);*/
}

document.getElementsByTagName("form")[0].addEventListener("submit", inputFile);
document.querySelector("#train").addEventListener("click", start);
//--------------

const float_MaxValue = 1; //Math.pow(2, 112)//3.4028235e38;//65500.0
const float_MinValue = -1; //-Math.pow(2,112)//-3.4028235e38;//-65500.0
var train = 0.01; //1.40129846432e-45

const surface = { name: "show.history live", tab: "Training" };

var m_predict;
var m_evaluate;

var history_train = new Array(256).fill(0);
var history_i = 0;

// Build and compile model.
async function start() {
  console.log("Backend", tf.getBackend());

  //adam / sgd / rmsprop / adamax
  //meanSquaredError / meanAbsoluteError
  //0.000001

  model.compile({
    optimizer: tf.train.adam(),
    // Just pass through rate and distortion as losses/metrics.
    loss: "meanSquaredError", //"binaryCrossentropy", //customLoss, //"meanSquaredError",//"categoricalCrossentropy",
    //metrics: pass_through_loss,
    metrics: ["acc"],
  });

  // Train model with fit().
  //await model.fitDataset(inputTensor, {
  await model.fit(inputTensor, inputTensor, {
    batchSize: 1,
    epochs: 1024,
    //learningRate: 1.40129846432e-45,
    //shuffle: true,
    //validationData: validation_data,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        history_train[history_i % 256] = log;
        console.log("Epoch: " + epoch + " Loss: " + log.loss);
        tfvis.show.history(surface, history_train, ["loss", "acc"]);
        history_i++;
      },
    },
  });

  model.summary();

  result = await model.evaluate(inputTensor, inputTensor, {
    batchSize: 1,
  });

  console.log("evaluate");
  result[0].print();
  result[1].print();

  // Run inference with predict().
  m_predict = await model.predict(inputTensor);
  console.log("predict", m_predict);
  m_predict.print();

  //const m_predict_rev = m_predict.mul(inputMax.sub(inputMin)).add(inputMin);

  const predictedPoints = m_predict.arraySync();

  let xPredic = [];
  let yPredic = [];

  for (
    let index = 0;
    index < predictedPoints[0].length && index < 10;
    index++
  ) {
    xPredic.push({
      x: predictedPoints[0][index],
      y: predictedPoints[0][index],
    });

    yPredic.push({
      x: inputTest[index],
      y: inputTest[index],
    });
  }

  console.log("difPredic", xPredic, yPredic);

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [xPredic, yPredic],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
//tf.setBackend("wasm");
