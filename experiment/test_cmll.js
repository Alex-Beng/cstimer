"use strict";

// Exhaustive check of Roux CMLL recognition: with both blocks solved (and the
// M slice / U layer free), every legal corner state must get a CMLL algorithm
// that solves the corners and keeps both blocks intact.  The corner case with
// a pure AUF left is handled as an AUF hint.
var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var ctx = {console: console, process: {env: {}}, module: {exports: {}}, Date: Date, Math: Math, JSON: JSON, self: {}};
vm.createContext(ctx);
vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
['isaac.js', 'utillib.js', 'mathlib.js', 'cmll.js', 'copilot.js'].forEach(function(f) {
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

function invParity(p) {
	var c = 0;
	for (var i = 0; i < p.length; i++)
		for (var j = i + 1; j < p.length; j++)
			if (p[i] > p[j]) c++;
	return c % 2;
}

function applyAlg(cc, tokens) {
	var ret = new mathlib.CubieCube();
	ret.init(cc.ca.slice(), cc.ea.slice());
	tokens.forEach(function(t) { ret.selfMoveStr(t); });
	return ret;
}

function cornersSolved(cc) {
	for (var i = 0; i < 4; i++) {
		if (cc.ca[i] != i) return false;
	}
	return true;
}

function f2bSolved(cc) {
	for (var c = 4; c < 8; c++) {
		if (cc.ca[c] != c) return false;
	}
	for (var e = 4; e < 12; e++) {
		if (e == 5 || e == 7) continue;
		if (cc.ea[e] != e * 2) return false;
	}
	return true;
}

function stateFor(perm, ori) {
	var ca = [];
	var ea = [];
	for (var i = 0; i < 8; i++) ca.push(i);
	for (i = 0; i < 12; i++) ea.push(i * 2);
	for (i = 0; i < 4; i++) ca[i] = perm[i] + 8 * ori[i];
	var pieces = [0, 1, 2, 3, 5, 7];
	var slots = [0, 1, 2, 3, 5, 7];
	var wanted = invParity(perm);
	for (var pi = 0; pi < PERM6.length; pi++) {
		if (invParity(PERM6[pi]) != wanted) continue;
		for (var k = 0; k < 6; k++) ea[slots[k]] = pieces[PERM6[pi][k]] * 2;
		break;
	}
	var cc = new mathlib.CubieCube();
	cc.init(ca, ea);
	return cc;
}

var PERM4 = permutations(4);
var PERM6 = permutations(6);

copilot.ensureCmllLut();
console.log('CMLL table', copilot.cmllLutSize());

var total = 0;
var auf = 0;
var bad = 0;
var samples = [];

for (var pi = 0; pi < PERM4.length; pi++) {
	for (var o0 = 0; o0 < 3; o0++) for (var o1 = 0; o1 < 3; o1++)
		for (var o2 = 0; o2 < 3; o2++) for (var o3 = 0; o3 < 3; o3++) {
			if ((o0 + o1 + o2 + o3) % 3) continue;
			var cc = stateFor(PERM4[pi], [o0, o1, o2, o3]);
			var facelet = cc.toFaceCube();
			var stage = copilot.getStage(facelet, 'roux');
			if (cornersSolved(cc)) {
				if (stage == 'cmll') {
					bad++;
					if (samples.length < 5) samples.push(['stage-after-cmll', facelet]);
				}
				continue;
			}
			total++;
			if (stage != 'cmll') {
				bad++;
				if (samples.length < 5) samples.push(['stage', stage, facelet]);
				continue;
			}
			var hint = copilot.getHint(facelet, {method: 'roux'});
			if (!hint || hint.kind != 'cmll' || hint.name == 'no-alg') {
				bad++;
				if (samples.length < 5) samples.push(['no-hint', facelet, hint && hint.name]);
				continue;
			}
			if (hint.name == 'AUF') auf++;
			var after = applyAlg(cc, hint.moves);
			if (!f2bSolved(after) || !cornersSolved(after) || copilot.getStage(after.toFaceCube(), 'roux') == 'cmll') {
				bad++;
				if (samples.length < 5) samples.push(['bad-alg', hint.name, hint.alg, facelet]);
				continue;
			}
			if (!copilot.validateAlg(facelet, hint.moves, hint)) {
				bad++;
				if (samples.length < 5) samples.push(['bad-validation', hint.name, hint.alg, facelet]);
			}
		}
}

// blocks not solved: the Roux hint must stay silent
var broken = new mathlib.CubieCube();
broken.selfMoveStr('R');
var brokenStage = copilot.getStage(broken.toFaceCube(), 'roux');
var brokenHint = copilot.getHint(broken.toFaceCube(), {method: 'roux'});
if (brokenStage != 'blocks' || !brokenHint || brokenHint.stage != 'blocks' || brokenHint.kind) {
	bad++;
	samples.push(['blocks', brokenStage, JSON.stringify(brokenHint)]);
}

console.log('corner states', total, 'AUF hints', auf);
console.log('samples', JSON.stringify(samples));
console.log(bad == 0 ? 'ALL OK' : ('FAILURES: ' + bad));
process.exitCode = bad == 0 ? 0 : 1;
