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

async function inputFile(event) {
  event.preventDefault();

  console.log(event.target[0].files[0]);

  inputTensor = await read_file_f32(event.target[0].files[0]);

  for (let index = 0; index < inputTensor.length && index < 10; index++) {
    inputTest.push(inputTensor[index]);
  }

  console.log("MAX", max, "MIN", min);
  const size = inputTensor.length;
  console.log("read_file_f32", inputTensor, size);

  const input = tf.input({ shape: [size] });

  const compress = tf.layers
    .dense({
      units: Math.round(size / 2),
      activation: "sigmoid",
    })
    .apply(input);

  const out = tf.layers.dense({ units: size }).apply(compress);

  model = tf.model({ inputs: input, outputs: out });
  model.summary();
  tfvis.show.modelSummary(surface, model);

  inputTensor = tf.tensor1d(inputTensor, "float32");
  //inputTensor = inputTensor.reshape([1, size]);
  console.log("inputTensor", inputTensor);

  inputMax = inputTensor.max();
  inputMin = inputTensor.min();
  console.log("inputMax");
  inputMax.print();
  console.log("inputMin");
  inputMin.print();
  const fileNIntArray = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

  console.log(fileNIntArray.print());
  console.log(fileNIntArray.max().print(), fileNIntArray.min().print());

  inputTensor = fileNIntArray.reshape([1, size]);
}

document.getElementsByTagName("form")[0].addEventListener("submit", inputFile);
document.querySelector("#train").addEventListener("click", start);
//--------------

const float_MaxValue = 1; //Math.pow(2, 112)//3.4028235e38;//65500.0
const float_MinValue = -1; //-Math.pow(2,112)//-3.4028235e38;//-65500.0
const train = 0.01;

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
    optimizer: tf.train.adam(train),
    loss: "meanSquaredError",
    metrics: ["acc"],
  });

  // Train model with fit().
  //await model.fitDataset(inputTensor, {
  await model.fit(inputTensor, inputTensor, {
    batchSize: 1, //inputTensor.size,
    epochs: 1024,
    learningRate: 0.0000000000000001,
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

  const m_predict_rev = m_predict.mul(inputMax.sub(inputMin)).add(inputMin);

  const predictedPoints = m_predict_rev.arraySync();

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
