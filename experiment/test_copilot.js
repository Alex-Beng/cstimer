"use strict";

// Verify the Copilot engine:
//   1. F2L next-pair hints always complete the pair, keep the cross and the
//      already solved pairs, and are truly minimal (brute force cross-check).
//   2. The ALLL LUT is complete (every entry solves its key) and we report how
//      much of the 62208 last-layer states it covers.
// Usage: node experiment/test_copilot.js [cases]

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var CASES = ~~process.argv[2] || 150;

function makeContext() {
	var ctx = {
		console: console,
		process: {env: {}},
		module: {exports: {}},
		Date: Date,
		Math: Math,
		JSON: JSON,
		setTimeout: setTimeout,
		self: {}
	};
	vm.createContext(ctx);
	vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/isaac.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
		.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/mathlib.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/alll.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/copilot.js'), 'utf8'), ctx);
	return ctx;
}

var ctx = makeContext();
var mathlib = ctx.mathlib;
var copilot = ctx.cubeCopilot;

var MOVE_NAMES = [];
for (var f = 0; f < 6; f++) {
	var face = 'URFDLB'.charAt(f);
	MOVE_NAMES.push(face, face + '2', face + "'");
}
var MOVE_CUBES = MOVE_NAMES.map(function(n) {
	var c = new mathlib.CubieCube();
	c.selfMoveStr(n);
	return c;
});

function mul(a, b) {
	var ret = new mathlib.CubieCube();
	mathlib.CubieCube.CubeMult(a, b, ret);
	return ret;
}

function applyMoves(cc, moves) {
	var cur = cc;
	for (var i = 0; i < moves.length; i++) {
		cur = mul(cur, MOVE_CUBES[moves[i]]);
	}
	return cur;
}

function applyAlg(cc, alg) {
	var cur = new mathlib.CubieCube();
	cur.init(cc.ca.slice(), cc.ea.slice());
	alg.split(/\s+/).filter(Boolean).forEach(function(t) {
		cur.selfMoveStr(t);
	});
	return cur;
}

function isPairSolved(cc, i) {
	var p = copilot.F2L_PAIRS[i];
	return (cc.ca[p.corner] & 7) == p.corner && (cc.ca[p.corner] >> 3) == 0 &&
		(cc.ea[p.edge] >> 1) == p.edge && (cc.ea[p.edge] & 1) == 0;
}

function crossSolved(cc) {
	for (var e = 4; e < 8; e++) {
		if ((cc.ea[e] >> 1) != e || (cc.ea[e] & 1) != 0) {
			return false;
		}
	}
	return true;
}

// ---- cross-solved position generator -------------------------------------
// BFS from the solved cube over the cross coordinate, storing a move path.
// Any random state can then be mapped to a cross-solved state by appending the
// path for its coordinate.

var CROSS_PATH = (function() {
	var CROSS_EDGES = [4, 5, 6, 7];
	// iterate over the cross pieces, not over slots: this coordinate is closed
	// under moves, so BFS distances are valid
	function coord(cc) {
		var out = [];
		for (var i = 0; i < CROSS_EDGES.length; i++) {
			var piece = CROSS_EDGES[i];
			for (var p = 0; p < 12; p++) {
				if ((cc.ea[p] >> 1) == piece) {
					out.push(p + ':' + (cc.ea[p] & 1));
					break;
				}
			}
		}
		return out.join('|');
	}
	var solved = new mathlib.CubieCube();
	var table = {};
	table[coord(solved)] = [];
	var frontier = [solved];
	var depth = 0;
	while (frontier.length && depth < 9) {
		var next = [];
		for (var i = 0; i < frontier.length; i++) {
			for (var m = 0; m < 18; m++) {
				var ns = mul(frontier[i], MOVE_CUBES[m]);
				var key = coord(ns);
				if (key in table) {
					continue;
				}
				table[key] = table[coord(frontier[i])].concat([m]);
				next.push(ns);
			}
		}
		frontier = next;
		depth++;
	}
	return {table: table, coord: coord};
})();

function crossSolvedFrom(cc) {
	var key = CROSS_PATH.coord(cc);
	var path = CROSS_PATH.table[key];
	if (!path) {
		return null;
	}
	// path goes solved -> key; to solve cc we need the reverse inverse
	var cur = cc;
	for (var i = path.length - 1; i >= 0; i--) {
		var m = path[i];
		var inv = Math.floor(m / 3) * 3 + (2 - m % 3);
		cur = mul(cur, MOVE_CUBES[inv]);
	}
	return cur;
}

function randomScramble(n) {
	var ret = [];
	var last = -1;
	for (var i = 0; i < n; i++) {
		var m;
		do {
			m = Math.floor(Math.random() * 18);
		} while (Math.floor(m / 3) == last);
		last = Math.floor(m / 3);
		ret.push(m);
	}
	return ret;
}

var failures = 0;
function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
}

// ---- brute force reference for the next-pair definition ------------------

function bruteShortestPair(base, maxDepth, solvedMask) {
	function check(state) {
		if (!crossSolved(state)) {
			return false;
		}
		var solvedNow = 0;
		for (var i = 0; i < 4; i++) {
			if (isPairSolved(state, i)) {
				solvedNow |= 1 << i;
			}
		}
		for (var i = 0; i < 4; i++) {
			if ((solvedMask & (1 << i)) && !(solvedNow & (1 << i))) {
				return false;
			}
		}
		return (solvedNow & ~solvedMask) != 0;
	}
	function dfs(state, depth, lastFace) {
		if (depth == 0) {
			return check(state);
		}
		for (var m = 0; m < 18; m++) {
			var face = Math.floor(m / 3);
			if (face == lastFace) {
				continue;
			}
			if (dfs(mul(state, MOVE_CUBES[m]), depth - 1, face)) {
				return true;
			}
		}
		return false;
	}
	for (var d = 1; d <= maxDepth; d++) {
		if (dfs(base, d, -1)) {
			return d;
		}
	}
	return 0;
}

// ---- 1. F2L --------------------------------------------------------------

var f2lTests = 0;
var f2lMoves = 0;
var minimalChecks = 0;
var attempts = 0;

while (f2lTests < CASES && attempts < 20000) {
	attempts++;
	var cc = new mathlib.CubieCube();
	cc = applyMoves(cc, randomScramble(20));
	var pos = crossSolvedFrom(cc);
	if (!pos) {
		continue;
	}
	var facelet = pos.toFaceCube();
	if (copilot.getStage(facelet) != 'f2l') {
		continue;
	}
	f2lTests++;
	var solvedMask = 0;
	for (var i = 0; i < 4; i++) {
		if (isPairSolved(pos, i)) {
			solvedMask |= 1 << i;
		}
	}
	var hint = copilot.solveNextPair(facelet, {maxDepth: 8});
	if (!hint) {
		fail('no F2L hint');
		continue;
	}
	f2lMoves += hint.moves.length;
	var after = applyAlg(pos, hint.alg);
	if (!crossSolved(after)) {
		fail('hint breaks cross: ' + hint.alg);
	}
	if (!isPairSolved(after, hint.pair)) {
		fail('hint does not solve its pair: ' + hint.alg);
	}
	for (var i = 0; i < 4; i++) {
		if ((solvedMask & (1 << i)) && !isPairSolved(after, i)) {
			fail('hint breaks solved pair ' + i + ': ' + hint.alg);
		}
	}
	if (hint.moves.length <= 5) {
		minimalChecks++;
		var ref = bruteShortestPair(pos, hint.moves.length - 1, solvedMask);
		if (ref) {
			fail('not minimal: hint ' + hint.moves.length + ' vs ' + ref);
		}
	}
}

console.log('F2L cases', f2lTests, 'avg moves', (f2lMoves / Math.max(f2lTests, 1)).toFixed(2),
	'minimality checks', minimalChecks);

// ---- 2. ALLL -------------------------------------------------------------

var stats = copilot.buildAlllLut();
var LL_STATES = 62208;
console.log('ALLL base algs', stats.entries, 'LUT states', stats.unique,
	'coverage', (stats.unique * 100 / LL_STATES).toFixed(2) + '%');

var lut = copilot.ensureAlllLut();
var verified = 0;
Object.keys(lut).forEach(function(key) {
	var entry = lut[key];
	var parts = key.split(',').map(Number);
	var cc = new mathlib.CubieCube();
	for (var i = 0; i < 4; i++) {
		cc.ca[i] = parts[i * 2] | (parts[i * 2 + 1] << 3);
	}
	for (var i = 4; i < 8; i++) {
		cc.ca[i] = i;
	}
	for (var i = 0; i < 4; i++) {
		cc.ea[i] = (parts[8 + i * 2] << 1) | parts[8 + i * 2 + 1];
	}
	for (var i = 4; i < 12; i++) {
		cc.ea[i] = i * 2;
	}
	var after = applyAlg(cc, entry.alg);
	if (after.toFaceCube() != mathlib.SOLVED_FACELET) {
		fail('LUT entry broken: ' + key + ' ' + entry.alg);
	}
	verified++;
});
console.log('LUT entries verified', verified);

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
