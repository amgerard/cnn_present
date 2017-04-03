var netA = new Network([2,4,4,1], 0.1, .5);
function train3() {
	var data = [];
	for (var i=0; i<trainingData.length; i++){
		var x = [trainingData[i]['input'][1],trainingData[i]['input'][2]];
		var y = trainingData[i]['output'];
		data.push(new DataSet(x, [y]));
	}
	netA.Train(data, 1000);
	W = [netA.Layers[1][0].Bias,
	     netA.Layers[1][0].InputSynapses[0].Weight,
	     netA.Layers[1][0].InputSynapses[1].Weight];
	console.log(W);
	return W;
}

var imageData = null;
var trainingData = [];

function drawPoint(x,y){
  
  var cls = document.getElementById("cbxBlue").checked;
  var xCalc = x/600.0; //(x-300)/600.0
  var yCalc = y/400.0; //(x-200)/400.0
  trainingData.push({input: [1, xCalc, yCalc], output: [cls ? 1 : 0]});
  w = train3();

  // get stuff
  var c = document.getElementById("myCanvas");
  var ctx = c.getContext("2d");
 
  ctx.clearRect(0, 0, c.width, c.height); 
  // reset
  //if (imageData !== null)    
  //  ctx.putImageData(imageData, 0, 0);
  
  for (var i = 0; i < 600; i+=7) {
    for (var j = 0; j < 400; j+=7) {
      var xi = i / 600.0; 
      var yj = j / 400.0;
      //res = math.dot(w, [1, xi, yj])
      res = netA.Compute([xi, yj])
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

// Add event listener for `click` events.
document.getElementById("myCanvas").addEventListener('click', function(event) {
    
    var pt = getMousePos(document.getElementById("myCanvas"), event);
    
    drawPoint(pt.x, pt.y);
    
    /*var x = event.pageX - elemLeft,
        y = event.pageY - elemTop;

    // Collision detection between clicked offset and element.
    elements.forEach(function(element) {
        if (y > element.top && y < element.top + element.height 
            && x > element.left && x < element.left + element.width) {
            alert('clicked an element');
        }
    });*/

}, false);