

// get rand
function randomInit(){ return Math.random()*2-1; }

// sigmoid activation
function sigmoid(x) { return x<-45.0 ? 0.0 : x>45.0 ? 1.0 : 1.0/(1.0+Math.pow(Math.E, -x)); }
function sigmoid_deriv(x) { return x * (1-x); }

// Dataset class
function DataSet(values, targets){
	this.Values = values;
	this.Targets = targets;
}

// Synapse class
function Synapse(inputNeuron, outputNeuron){

	// Synapse properties
	this.InputNeuron = inputNeuron;
	this.OutputNeuron = outputNeuron;
	this.Weight = randomInit();	
	this.WeightDelta = 0;	
}

// Neuron class
function Neuron(inputNeurons){

	// Neuron properties
	this.InputSynapses = [];
	this.OutputSynapses = [];
	this.Bias = randomInit();
	this.BiasDelta = 0;
	this.Gradient = 0;
	this.Output = 0;

	if (inputNeurons !== undefined){
		for (var i=0; i<inputNeurons.length; i++){
			var inNeuron = inputNeurons[i];
			var synapse = new Synapse(inNeuron, this);
			this.InputSynapses.push(synapse);
			inNeuron.OutputSynapses.push(synapse);
		}
	}
}

// Network class
function Network(layerSizes, learnRate, momentum){
	this.LearnRate = learnRate || 0.4;
	this.Momentum = momentum || 0.9;
	this.Layers = [];
	
	if (layerSizes){
		for (var i=0; i<layerSizes.length; i++){
			this.Layers.push([]);
			for (var j=0; j<layerSizes[i]; j++){
				var inputLayer = i===0 ? undefined : this.Layers[i-1];
				this.Layers[i].push(new Neuron(inputLayer));
			}	
		}
	}
}

// Neuron Implementation
/**
 * Take dot-product of prev layer outputs with weights and pass through activation
 * @return {Number} Result
 */
Neuron.prototype.CalcOutput = function(){
	/*var inputsTimesWeights = this.InputSynapses.map(s => s.Weight * s.InputNeuron.Output);
	var sumIn = inputsTimesWeights.reduce((i,j) => i + j);*/
	var sumIn = 0;
	for (var i=0; i<this.InputSynapses.length; i++){
		var synapse = this.InputSynapses[i];
		sumIn += synapse.Weight * synapse.InputNeuron.Output;
	}
	this.Output = sigmoid(sumIn + this.Bias);
	return this.Output;
}
Neuron.prototype.CalcError = function(target){
	return target-this.Output;
}
Neuron.prototype.CalcGrad1 = function(){
	var inputTimesWeights = this.OutputSynapses.map(s => s.OutputNeuron.Gradient * s.Weight);
	this.Gradient = inputTimesWeights.reduce((i,j) =>i+j) * sigmoid_deriv(this.Output);
	return this.Gradient;
}
Neuron.prototype.CalcGrad2 = function(target){
	this.Gradient = this.CalcError(target) * sigmoid_deriv(this.Output);
	return this.Gradient;
}
Neuron.prototype.CalcGrad = function(target){
	//console.log('b4: ' + this.Gradient);
	if (target !== undefined){ // output layer
		this.Gradient = this.CalcError(target) * sigmoid_deriv(this.Output);
	}
	else{ // hidden layers
		//var inputTimesWeights = this.OutputSynapses.map(s => s.OutputNeuron.Gradient * s.Weight);
		//this.Gradient = inputTimesWeights.reduce((i,j) =>i+j) * sigmoid_deriv(this.Output);
		var sumOut = 0;
		for (var i=0; i<this.OutputSynapses.length; i++){
			var synapse = this.OutputSynapses[i];
			sumOut += synapse.OutputNeuron.Gradient * synapse.Weight;
		}
		this.Gradient = sumOut * sigmoid_deriv(this.Output);
	}
	//console.log('af: ' + this.Gradient);
	return this.Gradient;
}
Neuron.prototype.UpdateWeights = function(learnRate, momentum, isj1, ss){
	var prevDelta = this.BiasDelta;
	this.BiasDelta = learnRate * this.Gradient;
	this.Bias += (this.BiasDelta + momentum * prevDelta);
	
	for (var i=0; i<this.InputSynapses.length; i++){
		var synapse = this.InputSynapses[i];
		prevDelta = synapse.WeightDelta;
		synapse.WeightDelta = learnRate * this.Gradient * synapse.InputNeuron.Output;
		synapse.Weight += (synapse.WeightDelta + momentum * prevDelta);
	}	
}

// Network Implementation
Network.prototype.Train = function(data, epochs){
	for (var i=0; i<epochs; i++){
		var errors = [];
		for (var j=0; j<data.length; j++){
			this.ForwardProp(data[j].Values);
			this.BackProp(data[j].Targets);
			errors.push(this.CalcError(data[j].Targets));
		}
		//console.log('err: ' + errors.reduce((a,b)=>a+b)/errors.length);
	}	
}
Network.prototype.ForwardProp = function(inputs){
	var i = 0;
	this.Layers[0].forEach(a => a.Output = inputs[i++]);
	for (var j=1; j<this.Layers.length; j++){
		this.Layers[j].forEach(a => a.CalcOutput());
	}
}
Network.prototype.BackProp = function(targets){
	var i = 0;
	//console.log(targets);
	var outIdx = this.Layers.length-1;
	//this.Layers[outIdx].forEach(a => a.CalcGrad2(targets[i++]));
	for (var j=0; j<this.Layers[outIdx].length; j++)
		this.Layers[outIdx][j].CalcGrad2(targets[j]);	

	for (var j=outIdx-1; j>=1; j--){
		this.Layers[j].forEach(a => a.CalcGrad1());
		//console.log('A ' + j);
	}
	for (var j=1; j<this.Layers.length; j++){
		for (var k=0; k<this.Layers[j].length; k++)
			this.Layers[j][k].UpdateWeights(this.LearnRate,this.Momentum, j===1, this.Layers[j][k].InputSynapses[0]);
	}
}
Network.prototype.Compute = function(inputs){
	this.ForwardProp(inputs);
	var outIdx = this.Layers.length-1;
	return this.Layers[outIdx].map(a => a.Output);
}
Network.prototype.CalcError = function(targets){
	var i=0;
	var absErrorByNeuron = this.Layers[this.Layers.length-1].map(a => Math.abs(a.CalcError(targets[i++])));
	return absErrorByNeuron.reduce((a,b)=>a+b);
}
Network.prototype.PrintNetwork = function(){
	for (var i=0; i<this.Layers.length; i++){ // each layer
		console.log('Layer ' + (i+1));	
		for (var j=0; j<this.Layers[i].length; j++){
			var O = this.Layers[i][j].Output;
			var B = this.Layers[i][j].Bias;
			console.log('  N' + (j+1) + ' [O=' + O + ', B=' + B + ']');			
			for (var k=0; k<this.Layers[i][j].OutputSynapses.length; k++){
				console.log('    S' + (k+1) + ' [W=' + this.Layers[i][j].OutputSynapses[k].Weight);			
			}
		}
	}
}
