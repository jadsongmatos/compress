async function read_file_32b(file) {
  const buffer = await file.arrayBuffer();
  let dataview = new DataView(buffer);
  let reuslt = new Int32Array(Math.ceil(file.size / 4));

  let num1 = 0;
  let num2 = 0;
  let num3 = 0;
  let num4 = 0;

  let tmp = 0;
  let index = 0;

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

    tmp = (num1 << 24) | (num2 << 16) | (num3 << 8) | num4;

    //console.log(tmp, index);

    reuslt[index] = tmp;
    index++;
  }
  return reuslt;
}

async function inputFile(event) {
  event.preventDefault();

  console.log(event.target[0].files[0]);

  const outIntArray = await read_file_32b(event.target[0].files[0]);

  console.log(outIntArray);

  let tmpInputT = tf.tensor1d(outIntArray, "int32");
  const inputMax = tmpInputT.max();
  const inputMin = tmpInputT.min();
  fileNIntArray = tmpInputT.sub(inputMin).div(inputMax.sub(inputMin));

  console.log(fileNIntArray.print());
  console.log(fileNIntArray.max().print(), fileNIntArray.min().print());

  //let tmpData = tf.data.array(await fileNIntArray.array()).batch(2);
  //console.log(tmpData);

  /*
  let out2Tensor = tf.data
    .array(tf.zeros([Math.ceil(size / 2)], "int32").arraySync())
    .batch(1);
  */

  //inputTensor = tf.data.zip({ xs: tmpData, ys: tmpData });
  inputTensor = fileNIntArray.reshape([Math.round(size / 2), 2]);

  start();
}

document.getElementsByTagName("form")[0].addEventListener("submit", inputFile);
//--------------

const float_MaxValue = 1; //Math.pow(2, 112)//3.4028235e38;//65500.0
const float_MinValue = -1; //-Math.pow(2,112)//-3.4028235e38;//-65500.0
const train = 0.01;

const surface = { name: "show.history live", tab: "Training" };

const input = tf.input({ shape: [2] });
const dense0 = tf.layers.dense({ units: 2 }).apply(input);
const dense1 = tf.layers.dense({ units: 2 }).apply(dense0);
const dense2 = tf.layers.dense({ units: 1 }).apply(dense1);
const dense3 = tf.layers.dense({ units: 2 }).apply(dense2);
const dense4 = tf.layers.dense({ units: 2 }).apply(dense3);

const model = tf.model({ inputs: input, outputs: dense4 });
model.summary();
tfvis.show.modelSummary(surface, model);

var m_predict;
var m_evaluate;

var history_train = [];

const testOut1 = tf.tensor2d([
  [3, 4],
  [7, 9],
  [-128, -1808],
]);

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
    batchSize: Math.round(size / 2),
    epochs: 128,
    //shuffle: true,
    //validationData: validation_data,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        history_train.push(log);
        console.log("Epoch: " + epoch + " Loss: " + log.loss);
        tfvis.show.history(surface, history_train, ["loss", "acc"]);
      },
    },
  });

  model.summary();

  result = await model.evaluate(testOut1, testOut1, {
    batchSize: 3,
  });

  console.log("evaluate");
  result[0].print();
  result[1].print();

  // Run inference with predict().
  m_predict = await model.predict(testOut1);
  console.log("predict", m_predict);
  m_predict[0].print();
  m_predict[1].print();

  const predictedPoints = m_predict[0].arraySync().map((val, i) => {
    return { x: val[0], y: val[1] };
  });

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [
        [
          { x: 3, y: 4 },
          { x: 7, y: 9 },
          { x: -128, y: -1808 },
        ],
        predictedPoints,
      ],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
tf.setBackend("wasm");
