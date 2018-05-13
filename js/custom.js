/* global $, document, MathJax */

/* net parameters */
function changeNumLayers() { //eslint-disable-line no-unused-vars
    var x = document.getElementById("num-layers").value;
    var net1 = document.getElementById("net-1");
    var net2 = document.getElementById("net-2");
    if (x === "1") {
        net1.style.display="block";
        net2.style.display="none";
    } else if (x === "2") {
        net1.style.display="none";
        net2.style.display="block";
    }
}

function changeNonLinearity() { //eslint-disable-line no-unused-vars
    var x = document.getElementById("nonlin-type").value;
    var nonLins = [document.getElementById("net-1-nonlin-1"),
                   document.getElementById("net-2-nonlin-1"),
                   document.getElementById("net-2-nonlin-2")];
    for (var i=0; i<nonLins.length; i++) {
        if (x === "linear") {
            nonLins[i].src = "images/linear.png";
        } else if (x === "sigmoid") {
            nonLins[i].src = "images/sigmoid.png";
        } else if (x === "softplus") {
            nonLins[i].src = "images/softplus.png";
        }
    }
}

/* control buttons */
function init() { //eslint-disable-line no-unused-vars
    if (isNet1()) {
        var net1mat1Jax = MathJax.Hub.getAllJax("net-1-mat-1");
        net1mat1 = randMatrix(3, 4, 2);
        MathJax.Hub.Queue(["Text", net1mat1Jax[0], matrix2Str(net1mat1)]);
    }
    if (isNet2()) {
        var net2mat1Jax = MathJax.Hub.getAllJax("net-2-mat-1");
        net2mat1 = randMatrix(3, 4, 2);
        MathJax.Hub.Queue(["Text", net2mat1Jax[0], matrix2Str(net2mat1)]);
        var net2mat2Jax = MathJax.Hub.getAllJax("net-2-mat-2");
        net2mat2 = randMatrix(3, 3, 2);
        MathJax.Hub.Queue(["Text", net2mat2Jax[0], matrix2Str(net2mat2)]);
    }
}

function nextSample() { //eslint-disable-line no-unused-vars
    var sample = globalData[Math.floor(Math.random()*globalData.length)];
    globalX = strVec2Vec(sample.slice(0, 4));
    globalYstar = [0.0, 0.0, 0.0];
    globalYstar[sample[4]] = 1.0;

    /* Net-1 sample */
    var net1mathJax = MathJax.Hub.getAllJax("net-1");
    MathJax.Hub.Queue(["Text", net1mathJax[0], vector2Str(globalX)]);
    MathJax.Hub.Queue(["Text", net1mathJax[net1mathJax.length-2], vector2Str(globalYstar)]);

    /* Net-2 sample */
    var net2mathJax = MathJax.Hub.getAllJax("net-2");
    MathJax.Hub.Queue(["Text", net2mathJax[0], vector2Str(globalX)]);
    MathJax.Hub.Queue(["Text", net2mathJax[net2mathJax.length-2], vector2Str(globalYstar)]);
}

function predict() { //eslint-disable-line no-unused-vars
    if (isNet1()) {
        globalZ1 = multMatVec(net1mat1, globalX);
        globalY1 = applyNonlin(globalZ1);

        var net1ymathJax = MathJax.Hub.getAllJax("net-1-y");
        MathJax.Hub.Queue(["Text", net1ymathJax[0], vector2Str(globalY1)]);
    } else if (isNet2()) {
        globalZ1 = multMatVec(net2mat1, globalX);
        globalY1 = applyNonlin(globalZ1);

        globalZ2 = multMatVec(net2mat2, globalY1);
        globalY2 = applyNonlin(globalZ2);

        var net2ymathJax = MathJax.Hub.getAllJax("net-2-y");
        MathJax.Hub.Queue(["Text", net2ymathJax[0], vector2Str(globalY2)]);
    }
}

function backprop() {
    if (isNet1()) {
        var d = globalY1 - globalYstar;
        var dy = sigmoid_prime(globalZ1);
        var 
        
    }
}

/* helper functions */
function isNet1() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "1";
}

function isNet2() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "2";
}

function applyNonlin(vector) {
    var type = document.getElementById("nonlin-type").value;
    var result = [];
    for (var i=0; i<vector.length; i++) {
        if (type === "linear") {
            result.push(vector[i]);
        } else if (type === "sigmoid") {
            result.push(sigmoid(vector[i]));
        } else if (type === "softplus") {
            result.push(softplus(vector[i]));
        }
    }
    return result;
}

function sigmoid(x) {
    return 1.0/(1.0+Math.exp(-x));
}

function softplus(x) {
    return Math.log(1.0 + Math.exp(x));
}

function sigmoid_prime(x) {
    return sigmoid(x)*(1.0-sigmoid(x));
}

function strVec2Vec(vector) {
    var v = [];
    for (var i=0; i<vector.length; i++) {
        v.push(parseFloat(vector[i]));
    }
    return v;
}

function vector2Str(vector) {
    var s = "\\begin{bmatrix}";
    for (var i=0; i<vector.length; i++) {
        s = s.concat(vector[i].toFixed(1));
        s = s.concat("\\\\");
    }
    s = s.concat("\\end{bmatrix}");
    return s;
}

function matrix2Str(matrix) {
    var s = "\\begin{bmatrix} ";
    for (var i=0; i<matrix.length; i++) {
        var delim = "";
        for (var j=0; j<matrix[i].length; j++) {
            s = s.concat(delim);
            s = s.concat(matrix[i][j].toFixed(1));
            delim = " & ";
        }
        s = s.concat(" \\\\ ");
    }
    s = s.concat(" \\end{bmatrix}");
    return s;
}

function randMatrix(height, width, scale) {
    var mat = [];
    for (var i=0; i<height; i++) {
        var row = [];
        for (var j=0; j<width; j++) {
            row.push((Math.random()-0.5)*scale);
        }
        mat.push(row);
    }
    return mat;
}

// Multiples a 2d matrix with a 1d vector, returns result as 1d vector
function multMatVec(mat, vec) {
    var result = [];
    for (var i=0; i<mat.length; i++) {
        var elem = 0;
        for (var j=0; j<mat[i].length; j++) {
            elem += mat[i][j]*vec[j];
        }
        result.push(elem);
    }
    return result;
}

// data
var globalData = [];
var iris2index = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2};

// net-1 params
var net1mat1;

// net-2 params
var net2mat1;
var net2mat2;

// net input/output values - needed for backprop.
var globalX;
var globalZ1;
var globalY1;
var globalZ2;
var globalY2;
var globalYstar;

/* Load data for training and testing */
$.ajax({
    url: "iris-data.txt",
    success: function(data) { 
        var lines = data.split(/\r?\n/);
        for (var i=0; i<lines.length; i++) {
            var line = lines[i].split(",");
            if (line.length === 5) {
                var sample = line.slice(0, 4);
                sample.push(iris2index[line[4]]);
                globalData.push(sample);
            }
        }
    }
});



