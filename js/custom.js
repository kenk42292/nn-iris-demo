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
        var net1mat1 = MathJax.Hub.getAllJax("net-1-mat-1");
        MathJax.Hub.Queue(["Text", net1mat1[0], matrix2Str(randMatrix(3, 4, 2))]);
    }
    if (isNet2()) {
        var net2mat1 = MathJax.Hub.getAllJax("net-2-mat-1");
        MathJax.Hub.Queue(["Text", net2mat1[0], matrix2Str(randMatrix(3, 4, 2))]);
        var net2mat2 = MathJax.Hub.getAllJax("net-2-mat-2");
        MathJax.Hub.Queue(["Text", net2mat2[0], matrix2Str(randMatrix(3, 3, 2))]);
    }
}

function nextSample() { //eslint-disable-line no-unused-vars
    var sample = globalData[Math.floor(Math.random()*globalData.length)];
    var sampleX = sample.slice(0, 4);
    var sampleY = [0.0, 0.0, 0.0];
    sampleY[sample[4]] = 1.0;

    /* Net-1 sample */
    var net1math = MathJax.Hub.getAllJax("net-1");
    MathJax.Hub.Queue(["Text", net1math[0], vector2Str(sampleX)]);
    MathJax.Hub.Queue(["Text", net1math[net1math.length-2], vector2Str(sampleY)]);

    /* Net-2 sample */
    var net2math = MathJax.Hub.getAllJax("net-2");
    MathJax.Hub.Queue(["Text", net2math[0], vector2Str(sampleX)]);
    MathJax.Hub.Queue(["Text", net2math[net2math.length-2], vector2Str(sampleY)]);
}

function predict() { //eslint-disable-line no-unused-vars
}

/* helper functions */
function isNet1() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "1";
}

function isNet2() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "2";
}

function vector2Str(vector) {
    var s = "\\begin{bmatrix}";
    for (var i=0; i<vector.length; i++) {
        s = s.concat(vector[i]);
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

var globalData = [];
var iris2index = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2};

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



