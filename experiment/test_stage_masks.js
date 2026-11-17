"use strict";

// Verify the corner-first / edge-first multi-phase patterns:
//   - a solved cube reaches the final phase (0)
//   - a twisted corner keeps "all corners solved" false but "all edges solved" true
//   - two swapped edges keep "all edges solved" false but "all corners solved" true
//   - both patterns report 2 steps and the expected phase order
// Usage: node experiment/test_stage_masks.js

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');

var ctx = {
	console: console,
	process: {env: {}},
	module: {exports: {}},
	Date: Date,
	Math: Math,
	JSON: JSON,
	self: {}
};
vm.createContext(ctx);
vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
vm.runInContext('var scramble_333={}, scramble_222={}, tools={};', ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/isaac.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
	.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/mathlib.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/cubeutil.js'), 'utf8'), ctx);

var mathlib = ctx.mathlib;
var cubeutil = ctx.cubeutil;

var failures = 0;
function check(name, cond) {
	if (!cond) {
		failures++;
		console.log('FAIL ' + name);
	}
}

function faceletOf(cc) {
	cc.ori = 0;
	return cc.toFaceCube();
}

var solved = mathlib.SOLVED_FACELET;
check('solved/corner', cubeutil.getProgress(solved, 'corner') == 0);
check('solved/edge', cubeutil.getProgress(solved, 'edge') == 0);

// a whole-cube rotation must not change the stage
var rot = new mathlib.CubieCube();
rot.ori = 5;
var fr = rot.toFaceCube(null, null, null, true);
check('rotated solved/corner', cubeutil.getProgress(fr, 'corner') == 0);
check('rotated solved/edge', cubeutil.getProgress(fr, 'edge') == 0);

var twist = new mathlib.CubieCube();
twist.ca[0] = (twist.ca[0] & 7) | (((twist.ca[0] >> 3) + 1) % 3 << 3);
var f1 = faceletOf(twist);
check('corner twist: corners unfinished', cubeutil.getProgress(f1, 'corner') == 2);
check('corner twist: edges still solved', cubeutil.getProgress(f1, 'edge') == 1);

var swap = new mathlib.CubieCube();
var tmp = swap.ea[0];
swap.ea[0] = swap.ea[1];
swap.ea[1] = tmp;
var f2 = faceletOf(swap);
check('edge swap: corners still solved', cubeutil.getProgress(f2, 'corner') == 1);
check('edge swap: edges unfinished', cubeutil.getProgress(f2, 'edge') == 2);

var mixed = new mathlib.CubieCube();
'R U F L D B R2 U2 F2'.split(' ').forEach(function(m) {
	mixed.selfMoveStr(m);
});
var f3 = faceletOf(mixed);
check('scramble: corner phase 2', cubeutil.getProgress(f3, 'corner') == 2);
check('scramble: edge phase 2', cubeutil.getProgress(f3, 'edge') == 2);

check('corner step count', cubeutil.getStepCount('corner') == 2);
check('edge step count', cubeutil.getStepCount('edge') == 2);
check('corner step order', cubeutil.getStepNames('corner').join(',') == 'edges,corners');
check('edge step order', cubeutil.getStepNames('edge').join(',') == 'corners,edges');

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
