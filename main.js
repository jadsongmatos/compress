var inputTensor;
var file;
var model;
var blocks_size;
var blocks

async function read_file_int16(file) {
  const buffer = await file.arrayBuffer();
  let buffer16 = new Int16Array(buffer);
  let reuslt = new Float32Array(buffer.byteLength / 2);

  for (let i = 0; i < buffer16.length; i++) {
    reuslt[i] = buffer16[i];
  }
  return reuslt;
}

async function inputFile(event) {
  event.preventDefault();

  file = event.target[0].files[0];
  console.log(event.target[0].files[0]);

  const size_16_b = Math.round(file.size / 2);
  console.log("size_16_b", size_16_b0);
  blocks_size = Math.round(size_16_b / 24);
  console.log("blocks_size", blocks_size);

  blocks = new Array(blocks_size).fill(0);

  const buffer = await event.target[0].files[0].arrayBuffer();
  let buffer16 = new Int16Array(buffer);
  let reuslt = new Float32Array(buffer.byteLength / 2);

  for (let i = 0; i < buffer16.length; i++) {
    reuslt[i] = buffer16[i];
  }
  return reuslt;
}

async function preLoad() {
  const size = inputTensor.length;
  console.log("read_file_int16", inputTensor, size);

  const input = tf.input({
    shape: [size],
    /// dtype: "string"
  });

  const units = Math.round(size / 64);
  const compress = tf.layers
    .dense({
      units: 1, //units <= 0 ? 1 : units,
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
}

document.getElementsByTagName("form")[0].addEventListener("submit", inputFile);
document.querySelector("#train").addEventListener("click", start);
//--------------

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

  model.compile({
    optimizer: tf.train.adam(),
    loss: "meanSquaredError", //"binaryCrossentropy", //customLoss, //"meanSquaredError",//"categoricalCrossentropy",
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
