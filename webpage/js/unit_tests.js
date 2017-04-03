function assert(condition, message) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

var inNeur = new Neuron();
var hidNeur = new Neuron([inNeur]);
var outNeur = new Neuron([hidNeur]);
console.log(hidNeur.CalcOutput());
console.log(hidNeur.CalcError(1));
console.log(hidNeur.CalcGrad(1));
console.log(sigmoid(1));

var net = new Network([2,1], 1.0, .9);

assert(net.Layers.length === 3, 'Layer Count Incorrect')
assert(net.Layers[0].length === 2, 'Layer 1 Neuron Count Incorrect')
assert(net.Layers[1].length === 2, 'Layer 2 Neuron Count Incorrect')
assert(net.Layers[2].length === 1, 'Layer 3 Neuron Count Incorrect')
assert(net.Layers[0][0].InputSynapses.length === 0, 'Layer 1 Neuron 1 Input Synapse Count Incorrect')
assert(net.Layers[0][1].InputSynapses.length === 0, 'Layer 1 Neuron 2 Input Synapse Count Incorrect')
assert(net.Layers[0][0].OutputSynapses.length === 2, 'Layer 1 Neuron 1 Output Synapse Count Incorrect')
assert(net.Layers[0][1].OutputSynapses.length === 2, 'Layer 1 Neuron 2 Output Synapse Count Incorrect')
assert(net.Layers[1][0].InputSynapses.length === 2, 'Layer 2 Neuron 1 Input Synapse Count Incorrect')
assert(net.Layers[1][1].InputSynapses.length === 2, 'Layer 2 Neuron 2 Input Synapse Count Incorrect')
assert(net.Layers[1][0].OutputSynapses.length === 1, 'Layer 2 Neuron 1 Output Synapse Count Incorrect')
assert(net.Layers[1][1].OutputSynapses.length === 1, 'Layer 2 Neuron 2 Output Synapse Count Incorrect')

var data = [];
data.push(new DataSet([0,0],[0]));
data.push(new DataSet([1,0],[1]));
data.push(new DataSet([0,1],[1]));
data.push(new DataSet([1,1],[1]));

//net.PrintNetwork();
console.log(net.Layers[2][0].Output)
net.ForwardProp(data[0].Values);
net.BackProp(data[0].Targets);
console.log(net.Layers[2][0].Output)
net.ForwardProp(data[0].Values);
net.BackProp(data[0].Targets);
console.log(net.Layers[2][0].Output)
net.ForwardProp(data[0].Values);
net.BackProp(data[0].Targets);
console.log(net.Layers[2][0].Output)
net.ForwardProp(data[0].Values);
net.BackProp(data[0].Targets);
console.log(net.Layers[2][0].Output)
net.ForwardProp(data[0].Values);
net.BackProp(data[0].Targets);
console.log(net.Layers[2][0].Output)
for (var i = 0; i < 20; i++){
	net.ForwardProp(data[0].Values);
	net.BackProp(data[0].Targets);
	console.log(net.Layers[2][0].Output)
}

net.PrintNetwork();
net.BackProp(data[0].Targets);
net.PrintNetwork();
net.ForwardProp(data[0].Values);
net.PrintNetwork();
net.BackProp(data[0].Targets);
net.PrintNetwork();


net.PrintNetwork();

net.Train(data, 10000);

console.log(net.Compute([0,0]));
console.log(net.Compute([0,1]));
console.log(net.Compute([1,0]));
console.log(net.Compute([1,1]));

net.PrintNetwork();