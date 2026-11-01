"use strict";

// Exact ALLL coverage by direct enumeration of every legal last-layer state
// (F2L solved): corner perm/orientation x edge perm/orientation with the
// parity constraint.  Then check which ones the LUT recognises.
var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var ctx = {console: console, process: {env: {}}, module: {exports: {}}, Date: Date, Math: Math, JSON: JSON, self: {}};
vm.createContext(ctx);
vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
['isaac.js', 'utillib.js', 'mathlib.js', 'alll.js', 'copilot.js'].forEach(function(f) {
	var c = fs.readFileSync(path.join(root, 'src/js/lib', f), 'utf8');
	if (f == 'utillib.js') c = c.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;');
	vm.runInContext(c, ctx);
});
var mathlib = ctx.mathlib;
var copilot = ctx.cubeCopilot;

function permutations(n) {
	var out = [];
	var arr = [];
	for (var i = 0; i < n; i++) arr.push(i);
	(function rec(k) {
		if (k == n) {
			out.push(arr.slice());
			return;
		}
		for (var i = k; i < n; i++) {
			var t = arr[k]; arr[k] = arr[i]; arr[i] = t;
			rec(k + 1);
			t = arr[k]; arr[k] = arr[i]; arr[i] = t;
		}
	})(0);
	return out;
}

function invCount(p) {
	var c = 0;
	for (var i = 0; i < p.length; i++)
		for (var j = i + 1; j < p.length; j++)
			if (p[i] > p[j]) c++;
	return c % 2;
}

var PERMS = permutations(4);
// orientation tuples with valid sums
var CORI = [];
for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) for (var c = 0; c < 3; c++) {
	var d = (3 - ((a + b + c) % 3)) % 3;
	CORI.push([a, b, c, d]);
}
var EORI = [];
for (var a = 0; a < 2; a++) for (var b = 0; b < 2; b++) for (var c = 0; c < 2; c++) {
	var d = (a + b + c) % 2;
	EORI.push([a, b, c, d]);
}
console.log('corner perms', PERMS.length, 'corner oris', CORI.length,
	'edge perms', PERMS.length, 'edge oris', EORI.length);

function keyOf(cp, co, ep, eo) {
	var ret = [];
	for (var i = 0; i < 4; i++) ret.push(cp[i], co[i]);
	for (var i = 0; i < 4; i++) ret.push(ep[i], eo[i]);
	return ret.join(',');
}

var stats = copilot.buildAlllLut();
var lut = copilot.ensureAlllLut();
console.log('LUT build:', JSON.stringify(stats));

var total = 0;
var covered = 0;
var unsolvedMiss = 0;
var missingSamples = [];
for (var ci = 0; ci < PERMS.length; ci++) {
	var cp = PERMS[ci];
	var cpar = invCount(cp);
	for (var cj = 0; cj < CORI.length; cj++) {
		var co = CORI[cj];
		for (var ei = 0; ei < PERMS.length; ei++) {
			var ep = PERMS[ei];
			if (invCount(ep) != cpar) continue;
			for (var ej = 0; ej < EORI.length; ej++) {
				var eo = EORI[ej];
				total++;
				var k = keyOf(cp, co, ep, eo);
				// build a real cube facelet for this last-layer state
				var cube = new mathlib.CubieCube();
				for (var i = 0; i < 4; i++) cube.ca[i] = cp[i] | (co[i] << 3);
				for (var i = 4; i < 8; i++) cube.ca[i] = i;
				for (var i = 0; i < 4; i++) cube.ea[i] = (ep[i] << 1) | eo[i];
				for (var i = 4; i < 12; i++) cube.ea[i] = i * 2;
				var res = copilot.solveLastLayer(cube.toFaceCube());
				if (res) covered++;
				else if (missingSamples.length < 6) missingSamples.push(k);
			}
		}
	}
}

console.log('legal last-layer states:', total);
console.log('covered by solveLastLayer:', covered, '(' + (covered * 100 / total).toFixed(3) + '%)');
console.log('missing sample:', JSON.stringify(missingSamples));

// validate every LUT entry against the exact key set
var badAlgs = 0;
var notLegal = 0;
var legal = {};
for (var ci = 0; ci < PERMS.length; ci++) {
	for (var cj = 0; cj < CORI.length; cj++) {
		for (var ei = 0; ei < PERMS.length; ei++) {
			if (invCount(PERMS[ei]) != invCount(PERMS[ci])) continue;
			for (var ej = 0; ej < EORI.length; ej++) {
				legal[keyOf(PERMS[ci], CORI[cj], PERMS[ei], EORI[ej])] = 1;
			}
		}
	}
}
Object.keys(lut).forEach(function(k) {
	if (!legal[k]) { notLegal++; return; }
	var parts = k.split(',').map(Number);
	var cc = new mathlib.CubieCube();
	for (var i = 0; i < 4; i++) cc.ca[i] = parts[i * 2] | (parts[i * 2 + 1] << 3);
	for (var i = 4; i < 8; i++) cc.ca[i] = i;
	for (var i = 0; i < 4; i++) cc.ea[i] = (parts[8 + i * 2] << 1) | parts[8 + i * 2 + 1];
	for (var i = 4; i < 12; i++) cc.ea[i] = i * 2;
	var after = new mathlib.CubieCube();
	after.init(cc.ca.slice(), cc.ea.slice());
	lut[k].alg.split(/\s+/).filter(Boolean).forEach(function(t) { after.selfMoveStr(t); });
	if (!after.isEqual(mathlib.CubieCube.SOLVED)) badAlgs++;
});
console.log('LUT keys illegal:', notLegal, '| LUT algs that fail:', badAlgs);
