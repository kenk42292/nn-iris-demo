/* global $, document */

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



