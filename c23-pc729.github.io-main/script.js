import {MnistData} from './data.js';
var saveButton, clearButton;
var rawImage;
var model;
	
function getModel() {
	model = tf.sequential();

	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

	model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});

	return model;
}

async function train(model, data) {
	const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];
	const container = { name: 'Model Training', styles: { height: '640px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 20,
		shuffle: true,
		callbacks: fitCallbacks
	});
}

function displayImages(imgArr) {
	const url = URL.createObjectURL(imgArr);
	rawImage.src = url;
}
    
function erase() {
	rawImage.src = "#";
}
    
function save() {
	document.getElementById("status").innerHTML = "Start Predicting...";
	var raw = tf.browser.fromPixels(rawImage,1);
	var resized = tf.image.resizeBilinear(raw, [28,28]);
	var tensor = resized.expandDims(0);
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
	document.getElementById("status").innerHTML = "Prediction result:";
	document.getElementById("result").innerHTML = pIndex;
	alert(pIndex);
}
    
function init() {
	const input = document.getElementById("finput");
	rawImage = document.getElementById('canvasimg');
	input.addEventListener("change", () => {
		let imageArray = [];
		const file = input.files;
		imageArray = file[0];
		console.log(imageArray);
		displayImages(imageArray);
	});
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}


async function run() {  
	const data = new MnistData();
	await data.load();
	const model = getModel();
	tfvis.show.modelSummary({name: 'Model Architecture'}, model);
	await train(model, data);
	init();
	alert("Training is done, try classifying your handwriting!");
	document.getElementById("caution").style = "color: white;";
}

document.addEventListener('DOMContentLoaded', run);



    
