'use strict';


function makeGensym() {
  var seq = 0;
  return function(prefix) {
    var result = prefix + seq;
    seq += 1;
    return result;
  };
}

module.exports = {
  makeGensym: makeGensym
};
