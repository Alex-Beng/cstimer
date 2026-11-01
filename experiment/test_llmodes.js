"use strict";

// Exhaustive check of the two-look last layer (OLL then PLL) over every legal
// last-layer state with F2L solved.  For every state the OLL hint must orient
// the layer, and the following PLL hint must solve it.  Also checks the 1LLL
// mode still solves everything.
var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var ctx = {console: console, process: {env: {}}, module: {exports: {}}, Date: Date, Math: Math, JSON: JSON, self: {}};
vm.createContext(ctx);
vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
['isaac.js', 'utillib.js', 'mathlib.js', 'alll.js', 'ollpll.js', 'copilot.js'].forEach(function(f) {
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
		if (k == n) { out.push(arr.slice()); return; }
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
function applyAlg(facelet, tokens) {
	var cc = new mathlib.CubieCube();
	cc.fromFacelet(facelet);
	tokens.forEach(function(t) { cc.selfMoveStr(t); });
	return cc.toFaceCube();
}

var PERMS = permutations(4);
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

copilot.warmUp();
console.log('OLL table', Object.keys(copilot.ensureOllLut()).length,
	'| PLL table', Object.keys(copilot.ensurePllLut()).length);

var total = 0;
var kinds = {oll: 0, pll: 0, alll: 0, auf: 0};
var bad = 0;
var samples = [];

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
				var cube = new mathlib.CubieCube();
				for (var i = 0; i < 4; i++) cube.ca[i] = cp[i] | (co[i] << 3);
				for (var i = 4; i < 8; i++) cube.ca[i] = i;
				for (var i = 0; i < 4; i++) cube.ea[i] = (ep[i] << 1) | eo[i];
				for (var i = 4; i < 12; i++) cube.ea[i] = i * 2;
				var facelet = cube.toFaceCube();

				var h1 = copilot.solveLastLayer(facelet, 'op');
				if (!h1) {
					bad++;
					if (samples.length < 5) samples.push(['no-hint', facelet]);
					continue;
				}
				kinds[h1.kind] = (kinds[h1.kind] || 0) + 1;
				var f2 = applyAlg(facelet, h1.moves);
				if (h1.kind == 'oll') {
					var h2 = copilot.solveLastLayer(f2, 'op');
					if (!h2) {
						bad++;
						if (samples.length < 5) samples.push(['no-pll', facelet, f2]);
						continue;
					}
					if (h2.kind == 'skip') {
						continue;
					}
					var f3 = applyAlg(f2, h2.moves);
					if (copilot.getStage(f3) != 'solved') {
						bad++;
						if (samples.length < 5) samples.push(['pll-fail', facelet, h2.alg]);
					}
				} else {
					if (copilot.getStage(f2) != 'solved') {
						bad++;
						if (samples.length < 5) samples.push(['fail', h1.kind, facelet, h1.alg]);
					}
				}

				var ha = copilot.solveLastLayer(facelet, 'alll');
				if (!ha || copilot.getStage(applyAlg(facelet, ha.moves)) != 'solved') {
					bad++;
					if (samples.length < 5) samples.push(['alll-fail', facelet, ha && ha.kind, ha && ha.alg,
						ha && copilot.getStage(applyAlg(facelet, ha.moves))]);
				}
			}
		}
	}
}

console.log('states', total, 'hint kinds', JSON.stringify(kinds));
console.log('samples', JSON.stringify(samples));
console.log(bad == 0 ? 'ALL OK' : ('FAILURES: ' + bad));
process.exitCode = bad == 0 ? 0 : 1;
