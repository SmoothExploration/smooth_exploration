var state_data;
var q_data;
var action_data;
var features;
var current_step = 0;
var current_state;
var settings;


jQuery(document).ready(function($) {

    function parseSettings() {
        labels = settings.split("__");
        $("#epsilonLabel").text(labels[2]);
        $("#alphaLabel").text(labels[3]);
        $("#labdaLabel").text(labels[4]);
        $("#kappaLabel").text(labels[5]);
        $("#actionInFeaturesLabel").text(labels[6]);
        $("#initializationLabel").text(labels[7].slice(0, -4));
        $("#totalStepsLabel").text(state_data.length);
        $("#currentStepLabel").text(current_step);
        $("#currentStateLabel").text(current_state);
    }

    function inArray(item, array) {
        len = array.length;

        for (x=0; x < len; x++) {
            if (item === array[x]) {
                return true;
            }
        }
        return false;
    }

    function updateUI() {
        state = state_data[current_step];
        q = q_data[current_step];
        action = q_data[current_step];

        var size = q.length;
        $("#table-body").html("");
        var tableData = ""
        // for (i=0; i < size; i++) {
        //     if (i === state) {
        //         tableData += "<tr id='currentState'><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
        //     } else {
        //         tableData += "<tr><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
        //     }
        // }
        for (i=50; i >= 0; i--) {
            if (i === state) {
                tableData += "<tr id='currentState'><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
            } else {
                tableData += "<tr><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
            }
        }
        for (i=99; i>=50; i--) {
            if (i === state) {
                tableData += "<tr id='currentState'><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
            } else {
                tableData += "<tr><td>" + i + "</td><td>" + q[i][0] + "</td><td>" + q[i][1] + "</td></tr>";
            }
        }
          
        $("#table-body").html(tableData);
        $('html,body').animate({
            scrollTop: $("#currentState").offset().top - 300
        }, 200);
        parseSettings();
    }

    function updateFuncUI() {
        state = state_data[current_step];
        q = q_data[current_step];
        action = q_data[current_step];
        f = features[current_step]

        features_array = [];

        for (i=0; i < f.length; i++) {
            if (f[i] === true) {
                features_array.push(i);
            }
        }

        current_state = state;

        var size = q[0].length;
        $("#table-body").html("");
        var tableData = ""
        for (i=0; i < size; i++) {
            if (inArray(i, features_array)) {
                tableData += "<tr id='currentState'><td>" + i + "</td><td>" + q[0][i] + "</td><td>" + q[1][i] + "</td></tr>";
            } else {
                tableData += "<tr><td>" + i + "</td><td>" + q[0][i] + "</td><td>" + q[1][i] + "</td></tr>";
            }
        }
        $("#table-body").html(tableData);
        $('html,body').animate({
            scrollTop: $("#currentState").offset().top - 300
        }, 200);
        parseSettings();
    }

    function getData(dataset, func){
        // if (dataset === "") {
        //     dataset = 'sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat'
        // }
        var url = "/data";
        if (func) {
            url = "/func_data";
        }
        console.log(dataset);
        jQuery.ajax({
          url: url,
          type: 'POST',
          contentType: 'application/json',
          dataType: 'json',
          data: JSON.stringify({run: dataset}),
          complete: function(xhr, textStatus) {
            //called when complete
          },
          success: function(data, textStatus, xhr) {
            //called when successful
            console.log(textStatus);
            console.log(data);
            state_data = data["states"];
            action_data = data["actions"];
            q_data = data["q_values"];
            features = data["features"]
            // console.log(state_data);
            // console.log(action_data);
            // console.log(q_data);
            settings = dataset;
            if (func) {
                updateFuncUI();
            } else {
                updateUI();
            }

          },
          error: function(xhr, textStatus, errorThrown) {
            //called when there is an error
          }
        });
    }

    $("#test").click(function() {
        var data = getData();
        // console.log(state_data);
    });

    $("#next").click(function() {
        current_step++;
        updateUI();
    });

    $("#previous").click(function() {
        current_step--;
        updateUI();
    });
    

    $(".dropdown-menu-func a").click(function() {
        // console.log('here');
        // console.log(this.text);
        var filename = this.text;
        // parseSettings(filename);
        current_step = 0;
        getData(filename, true);
        var source_filename = "static/img/" + filename + ".png "
        $("#graph").attr('src', source_filename)
    });

    $(".dropdown-menu a").click(function() {
        // console.log('here');
        // console.log(this.text);
        var filename = this.text;
        // parseSettings(filename);
        current_step = 0;
        getData(filename, false);
        var source_filename = "static/img/" + filename + ".png "
        $("#graph").attr('src', source_filename)
    });

    $("#nextFunc").click(function() {
        current_step++;
        updateFuncUI();
    });

    $("#previousFunc").click(function() {
        current_step--;
        updateFuncUI();
    });

    $("#step-button-tab").click(function() {
        current_step = $("#choose-state").val();
        updateUI();
    });

    $("#step-button-func").click(function() {
        current_step = $("#choose-state").val();
        updateFuncUI();
    });


});