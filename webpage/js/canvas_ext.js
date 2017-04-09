var LEARNING_RATE = 0.4;
var NUM_EPOCHS = 100;
var ARCH = [2,4,4,1];
var NET = new Network(ARCH, LEARNING_RATE);

var imageData = null;
var trainingData = [];

// get canvas
var c = document.getElementById("myCanvas");
var ctx = c.getContext("2d");

function train() {
	var data = [];
	for (var i=0; i<trainingData.length; i++){
		var x1 = trainingData[i]['input'][1];
		var x2 = trainingData[i]['input'][2];
		var x = [x1, x2].concat(getInputs() == 4 ? [x1*x1,x2*x2] : []);
		var y = trainingData[i]['output'];
		//console.log(x);
		data.push(new DataSet(x, [y]));
	}
	NET.Train(data, NUM_EPOCHS);
}
function restart() {
  	ctx.clearRect(0, 0, c.width, c.height); 
	trainingData = [];
        printNumEpochs(0);
	NET = new Network(ARCH, LEARNING_RATE);
}
function getInputs(cbInputs) {
	if (cbInputs === undefined)
		cbInputs = document.getElementById("cbInputs");
	return cbInputs.checked ? 4 : 2;
}
function getHidden() {
	var hidden = document.getElementById("hidden").value;
	return hidden.split(" ").map(Number).filter(Boolean);
}
function hiddenChanged() {
	networkChanged(getInputs());
}
function networkChanged(numIn) {
	ARCH = [numIn].concat(getHidden()).concat([1]);
	printNumEpochs(0);
	//console.log(ARCH);
	NET = new Network(ARCH, LEARNING_RATE);
}
function inputsChanged(cbInputs) {
	var numIn = getInputs(cbInputs);
	var tbxInputs = document.getElementById("tbxInputs");
	tbxInputs.innerHTML = "Inputs: " + numIn;
	var pInputs = document.getElementById("pInputs");
	pInputs.innerHTML = numIn == 2 ? "[x1  x2]" : "[x1  x2  x1<sup>2</sup>  x2<sup>2</sup>]";
	networkChanged(numIn);
}

function drawPoint(x,y){
  
  var cls = document.getElementById("cbxBlue").checked;
  var xCalc = x/600.0;
  var yCalc = y/400.0;
  trainingData.push({input: [1, xCalc, yCalc], output: [cls ? 1 : 0]});
  drawPoints();
}  

function printNumEpochs(val){
  var numEpochsEl = document.getElementById("printNumEpochs");
  numEpochsEl.innerHTML = val !== undefined ? val : parseInt(numEpochsEl.innerHTML) + 100;
  //console.log(numEpochsEl.value)
}

function drawTrain(){
  if (trainingData.length === 0)
    return;
  printNumEpochs();
  w = train();
 
  ctx.clearRect(0, 0, c.width, c.height); 
  // reset
  //if (imageData !== null)    
  //  ctx.putImageData(imageData, 0, 0);
  var numIn = getInputs();
  for (var i = 0; i < 600; i+=7) {
    for (var j = 0; j < 400; j+=7) {
      var xi = i / 600.0; 
      var yj = j / 400.0;
      //res = math.dot(w, [1, xi, yj])
      res = NET.Compute([xi, yj].concat(numIn == 4 ? [xi*xi,yj*yj] : []))
      ctx.fillStyle = res >= 0.5 ? '#9999FF' : '#FF9999';
      ctx.fillRect(i,j,i+7,j+7);
      /*ctx.beginPath();
      ctx.arc(i, j, 6, 0, 2 * Math.PI);
      //ctx.fillStyle = cls ? 'blue' : 'red';
      // res < 0.6 && res > 0.4 ? 'black' : 
      ctx.fill();*/
      //ctx.stroke();
    }
  }
  drawPoints();
}

function drawPoints(){
  //
  for (var i=0; i<trainingData.length; i++){
  ctx.beginPath();
  //ctx.arc(100, 75, 5, 0, 2 * Math.PI);
  ctx.arc(trainingData[i].input[1]*600, trainingData[i].input[2]*400, 5, 0, 2 * Math.PI);
  ctx.fillStyle = trainingData[i].output[0] ? 'blue' : 'red';
  ctx.fill();
  ctx.stroke();
  }

  // save the state of  the canvas here
  //imageData = ctx.getImageData(0,0,c.width,c.height);
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top
    };
}

// canvas click
document.getElementById("myCanvas").addEventListener('click', function(event) {
    var pt = getMousePos(document.getElementById("myCanvas"), event);
    drawPoint(pt.x, pt.y);
}, false);

// num epochs changed
document.getElementById("selNumEpochs").addEventListener('onchange', function (event) {
    var numEpochs = document.getElementById("selNumEpochs").value;
    NUM_EPOCHS = parseInt(numEpochs);
}, false);

// activation function changed
document.getElementById("selActivation").addEventListener('onchange', function (event) {
    var act = document.getElementById("selActivation").value;
    if (act === "Sigmoid")
        ACTIVATION = new Sigmoid();
    if (act === "Tanh")
        ACTIVATION = new Tanh();
    if (act === "Relu")
        ACTIVATION = new Relu();
}, false);

// learning rate changed
document.getElementById("inLearningRate").addEventListener('onchange', function (event) {
    var rate = document.getElementById("inLearningRate").value;
    LEARNING_RATE = parseFloat(rate);
}, false);
