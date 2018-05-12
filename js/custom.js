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

/* helper functions */
function isNet1() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value() === "1";
}

function isNet2() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value() === "2";
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
