/* global $, document */

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
    console.log(x);
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

$(document).ready(function() {
    'use strict';
    $('#next-input').click(function(event) { //eslint-disable-line no-unused-vars
        'use strict';
        // remove 'active' from all li
        $('.navbar-nav li a').parent().removeClass("active");
        // add 'active' to specific li
        $(this).parent().addClass("active");
    })
});

$(document).ready(function() {
    'use strict';
    $('#next-input').click(function(event) { //eslint-disable-line no-unused-vars
        'use strict';
        // remove 'active' from all li
        $('.navbar-nav li a').parent().removeClass("active");
        // add 'active' to specific li
        $(this).parent().addClass("active");
    })
});



