var size = 0;
var fileInt32Array;
var fileNInt32Array;
var inputTensor;
var outIntArray;

async function inputFile(event) {
  event.preventDefault();

  console.log(event.target[0].files[0]);

  let tmpBuffer = await event.target[0].files[0].arrayBuffer();
  let dataview = new DataView(tmpBuffer);
  size = tmpBuffer.byteLength / 4;

  outIntArray = new Int32Array(size);

  for (var i = 0; i < outIntArray.length; i++) {
    outIntArray[i] = dataview.getInt16(i * 4);
  }

  console.log(outIntArray);

  let tmpInputT = tf.tensor1d(outIntArray);
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
  result[0].print()
  result[1].print()

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
