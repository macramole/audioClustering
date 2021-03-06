var margin = {top: 0, right: 0, bottom: 0, left: 0},
    width = $(window).width() * 0.99 - margin.left - margin.right,
    height = $(window).height() * 0.99 - margin.top - margin.bottom;

// Audio
var audio = $("#audio")[0];

var DOT_SIZE = 4;

var cantClusters = null;
var nombreClusters = ["Platos / Agudos", "Bombos / Graves", "Redoblantes / Congas / Medios altos", "Toms bajos / Medios graves", "Campanas"];

var shiftPressed = false;

var cantPerCluster;

var mostrarEspectrograma = false;
$("#mostrarEspectrograma").click( function() {
    mostrarEspectrograma = $(this).is(":checked");
});

//zoom
var zoom = d3.behavior.zoom()
    .scaleExtent([1,10])
    .on("zoom", function() {
        svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    })
    ;
var drag = d3.behavior.drag()
    .origin(function(d) { return d; })
    .on("dragstart", function(d) {
      d3.event.sourceEvent.stopPropagation();
      d3.select(this).classed("dragging", true);
    })
    .on("drag", function(d) {
      d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
    })
    .on("dragend", function(d) {
      d3.select(this).classed("dragging", false);
    });


/*
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */

// setup x
var xValue = function(d) { return d.x;}, // data -> value
    xScale = d3.scale.linear().range([0, width]), // value -> display
    xMap = function(d) { return xScale(xValue(d));}, // data -> display
    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

// setup y
var yValue = function(d) { return d.y;}, // data -> value
    yScale = d3.scale.linear().range([height, 0]), // value -> display
    yMap = function(d) { return yScale(yValue(d));}, // data -> display
    yAxis = d3.svg.axis().scale(yScale).orient("left");

// setup fill color
var cValue = function(d) { return d.cluster;},
    color = d3.scale.category10();

// add the graph canvas to the body of the webpage
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .call(zoom)
    ;

svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .style("pointer-events", "all")
    ;


// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);


d3.tsv("audioClusteringResult.tsv", function(error, data) {

  // change string (from CSV) into number format
  var maxCluster = 0;

  cantPerCluster = [];

  data.forEach(function(d) {
      d.x = +d.x;
      d.y = +d.y;
      d.cluster = +d.cluster;

      if ( !cantPerCluster[d.cluster] ) {
          cantPerCluster[d.cluster] = 1;
      } else {
          cantPerCluster[d.cluster]++;
      }

      if ( d.cluster > maxCluster ) {
          maxCluster = d.cluster;
      }

      d.file = d.file.replace(".wav", ".mp3");
  });

  cantClusters = maxCluster;

  renderClustersItems();

  // don't want dots overlapping axis, so add in buffer to data domain
  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

  // x-axis
  // svg.append("g")
  //     .attr("class", "x axis")
  //     .attr("transform", "translate(0," + height + ")")
  //     .call(xAxis)
  //   .append("text")
  //     .attr("class", "label")
  //     .attr("x", width)
  //     .attr("y", -6)
  //     .style("text-anchor", "end")
  //     .text("X");
  //
  // // y-axis
  // svg.append("g")
  //     .attr("class", "y axis")
  //     .call(yAxis)
  //   .append("text")
  //     .attr("class", "label")
  //     .attr("transform", "rotate(-90)")
  //     .attr("y", 6)
  //     .attr("dy", ".71em")
  //     .style("text-anchor", "end")
  //     .text("Y");

  // draw dots
  svg.selectAll(".dot")
      .data(data)
    .enter().append("circle")
      .attr("class", "dot")
      .attr("r", DOT_SIZE)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .attr("data-cluster", function(d) { return d.cluster })
      .style("fill", function(d) { return color(cValue(d));})
      .on("click", function(d) {
          if ( !d3.select(this).classed("filtered") ) {
              audio.src = "sounds.mp3/" + d.file;
              audio.play();
              d3.select(this).attr("r", DOT_SIZE * 3).transition().attr("r", DOT_SIZE);
          }
      })
      .on("mouseover", function(d) {
          d3.select(this).style("stroke", "black");

          if ( mostrarEspectrograma && !d3.select(this).classed("filtered")  ) {
              tooltip.transition()
              .duration(200)
              .style("opacity", 1);
              //   tooltip.html("Filename:" + d.file)
              tooltip.html("<div>" + d.file + "</div><img src='espectrogramas/" + data.indexOf(d) + ".png' />")
            //   .style("left", (d3.event.pageX + 5) + "px")
            //   .style("top", (d3.event.pageY - 28) + "px");
              ;
          }
      })
      .on("mouseout", function(d) {
          d3.select(this).style("stroke", "");

          if ( mostrarEspectrograma ) {
              tooltip.transition()
                   .duration(500)
                   .style("opacity", 0)
                   ;
          }
      });

  // draw legend
  // var legend = svg.selectAll(".legend")
  //     .data(color.domain())
  //   .enter().append("g")
  //     .attr("class", "legend")
  //     .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
  //
  // // draw legend colored rectangles
  // legend.append("rect")
  //     .attr("x", width - 18)
  //     .attr("width", 18)
  //     .attr("height", 18)
  //     .style("fill", color);
  //
  // // draw legend text
  // legend.append("text")
  //     .attr("x", width - 24)
  //     .attr("y", 9)
  //     .attr("dy", ".35em")
  //     .style("text-anchor", "end")
  //     .text(function(d) { return d;})
});

function filterCluster(cluster) {
    $dots = $(".dot[data-cluster='" + cluster + "']");
    if ( $dots.eq(0).hasClass("filtered") ) {
        $dots.removeClass("filtered");
    } else {
        $dots.addClass("filtered");
    }
}

function renderClustersItems() {
    // $("body").mousedown(function(e) {
    //     shiftPressed = e.shiftKey;
    // })
    for( var i = 0 ; i <= cantClusters ; i++ ) {
        var $container = $("<div class='cluster'>");
        var $check = $("<input type='checkbox' />").val(i).attr("id", "chkCluster" + i).attr("checked", true);
        // var $item = $("<label for='" + "chkCluster" + i + "'>").text( nombreClusters[i] + " (" + i + ")").attr("data-cluster", i);
        var $item = $("<label for='" + "chkCluster" + i + "'>").text( nombreClusters[i] + " (" + cantPerCluster[i] + ")" ).attr("data-cluster", i);

        $item.css("background-color", color(i));
        $container.css("border-color", color(i));

        $container.append($check).append($item);
        $("#clusterItems").append($container);

        $check.change(function() {
            filterCluster ( $(this).val() );
            // if ( shiftPressed ) {
            //     console.log("asd");
            // }
        });

        $item.mouseover( function() {
            if ( mostrarEspectrograma ) {
                tooltip.transition().duration(200)
                .style("opacity", 1);
                tooltip.html("<img src='espectrogramasCluster/" + $(this).attr("data-cluster") + ".png' />")
            }
        });
        $item.mouseleave( function() {
            if ( mostrarEspectrograma ) {
                tooltip
                .transition()
                .duration(500)
                .style("opacity", 0)
                ;
            }
        });
    }
}
