/* ******** MathJs ******** */

//<script src="https://cdnjs.cloudflare.com/ajax/libs/synaptic/1.0.8/synaptic.min.js"></script>
//<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/3.9.3/math.min.js"></script>

// perceptron
/*
function sigmoid(z) {
	//return 1 / ( 1 + math.pow(math.e, X));
	return z.map(function(x){return 1/(1+math.pow(math.e,-x));})
}
*/
function cost(theta, X, y) {
	n = X._size[0];
}

function train_perceptron() {

	W = math.random([3], -0.1, 0.1);  // generate a matrix with random numbers
	learning_rate = .1;
	//X = math.matrix(trainingData.map(function(x){return x["input"];}));

	for(var epoch = 0; epoch < 10000; epoch++) {
		var index = Math.floor(Math.random() * (trainingData.length));
		x = trainingData[index]['input'];
		y = trainingData[index]['output'];
		res = math.dot(x,W);
		error = y - (res >= 0);
		W = math.add(W, math.multiply(x, learning_rate * error));
	}
	
	console.log(W);
	return W;
}