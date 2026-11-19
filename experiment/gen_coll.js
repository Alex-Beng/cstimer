"use strict";

// Generate src/js/lib/coll.js from a COLL catalogue.
//
// The catalogue is an array of {name, algs} entries (names such as "AS 1" or
// "Pi 3").  Algorithms are usually given with a leading regrip (`y ...`); the
// copilot works in a fixed frame, so whole-cube rotations are cancelled by a
// pre-rotation when necessary.  One algorithm per case is kept.
//
// Usage:
//   node experiment/gen_coll.js path/to/coll_cases.json [--write]

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var srcFile = process.argv[2] && process.argv[2].charAt(0) != '-' ? process.argv[2] : null;
var outFile = path.resolve(root, 'src', 'js', 'lib', 'coll.js');
var write = process.argv.indexOf('--write') >= 0;

if (!srcFile) {
	throw new Error('usage: node experiment/gen_coll.js path/to/coll_cases.json [--write]');
}

function loadMathlib() {
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
	vm.runInContext('var $={};', ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/isaac.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
		.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/mathlib.js'), 'utf8'), ctx);
	return ctx.mathlib;
}

var mathlib = loadMathlib();
var AUF = ['', 'U', 'U2', "U'"];
var MOVE_RE = /^([URFDLB]w?|[EMSxyz])(2|'|2')?$/;

function tokensOf(str) {
	var raw = str.split(/\s+/).filter(function(t) {
		return !!t;
	});
	var out = [];
	for (var i = 0; i < raw.length; i++) {
		var m = /^([urfdlb])(2|'|2')?$/.exec(raw[i]);
		if (m) {
			out.push(m[1].toUpperCase() + 'w' + (m[2] || ''));
		} else {
			out.push(raw[i]);
		}
	}
	return out;
}

function validTokens(ts) {
	return ts.length > 0 && ts.every(function(t) {
		return MOVE_RE.test(t);
	});
}


function simplify(ts) {
	var out = [];
	for (var i = 0; i < ts.length; i++) {
		var m = /^([URFDLB]w?|[EMSxyz])(2|'|2')?$/.exec(ts[i]);
		if (!m) {
			out.push(ts[i]);
			continue;
		}
		var base = m[1];
		var pow = powOf(m[2] || '');
		if (out.length) {
			var last = /^([URFDLB]w?|[EMSxyz])(2|'|2')?$/.exec(out[out.length - 1]);
			if (last && last[1] == base) {
				pow = (pow + powOf(last[2] || '')) % 4;
				out.pop();
			}
		}
		if (pow) {
			out.push(base + (pow == 1 ? '' : (pow == 2 ? '2' : "'")));
		}
	}
	return out;
}

function powOf(mod) {
	if (mod == "'") {
		return 3;
	}
	if (mod == '2' || mod == "2'") {
		return 2;
	}
	return 1;
}

function movesAppliedBy(token, ori) {
	var m = /^([URFDLB])(w?)(2|'|2')?$/.exec(token);
	if (m) {
		var axis = 'URFDLB'.indexOf(m[1]);
		var pow = powOf(m[3] || '');
		var faceIdx = m[2] ? (axis + 3) % 6 : axis;
		return [mathlib.CubieCube.rotMulM[ori][faceIdx * 3 + pow % 4 - 1]];
	}
	var s = /^([EMS])(2|'|2')?$/.exec(token);
	if (s) {
		var saxis = {S: 2, E: 3, M: 4}[s[1]];
		var spow = powOf(s[2] || '');
		return [
			mathlib.CubieCube.rotMulM[ori][saxis * 3 + (4 - spow) % 4 - 1],
			mathlib.CubieCube.rotMulM[ori][(saxis + 3) % 6 * 3 + spow % 4 - 1]
		];
	}
	return [];
}

function inverseState(tokens) {
	var probe = new mathlib.CubieCube();
	var oriBefore = new Array(tokens.length);
	for (var i = 0; i < tokens.length; i++) {
		oriBefore[i] = probe.ori;
		probe.selfMoveStr(tokens[i]);
	}
	var cc = new mathlib.CubieCube();
	var tmp = new mathlib.CubieCube();
	for (var i = tokens.length - 1; i >= 0; i--) {
		var applied = movesAppliedBy(tokens[i], oriBefore[i]);
		for (var k = applied.length - 1; k >= 0; k--) {
			var inv = ~~(applied[k] / 3) * 3 + (2 - applied[k] % 3);
			mathlib.CubieCube.CubeMult(cc, mathlib.CubieCube.moveCube[inv], tmp);
			cc.init(tmp.ca, tmp.ea);
		}
	}
	cc.ori = 0;
	return cc;
}

function isF2LDone(cc) {
	for (var e = 4; e < 8; e++) {
		if ((cc.ea[e] >> 1) != e || (cc.ea[e] & 1) != 0) {
			return false;
		}
	}
	var pairs = [[4, 8], [5, 9], [6, 10], [7, 11]];
	for (var i = 0; i < pairs.length; i++) {
		var c = pairs[i][0];
		var e = pairs[i][1];
		if ((cc.ca[c] & 7) != c || (cc.ca[c] >> 3) != 0 ||
			(cc.ea[e] >> 1) != e || (cc.ea[e] & 1) != 0) {
			return false;
		}
	}
	return true;
}

function edgesOriented(cc) {
	for (var i = 0; i < 4; i++) {
		if (cc.ea[i] & 1) {
			return false;
		}
	}
	return true;
}

function cornersSolved(cc) {
	for (var i = 0; i < 4; i++) {
		if (cc.ca[i] != i) {
			return false;
		}
	}
	return true;
}

function rotationsToTry() {
	var ret = [[]];
	for (var r = 1; r < 24; r++) {
		var str = mathlib.CubieCube.rot2str[r];
		if (str) {
			ret.push(tokensOf(str));
		}
	}
	return ret;
}

var ROTATIONS = rotationsToTry();

function usableAlg(ts) {
	var best = null;
	for (var r = 0; r < ROTATIONS.length; r++) {
		var seq = simplify(ROTATIONS[r].concat(ts));
		var probe = new mathlib.CubieCube();
		for (var i = 0; i < seq.length; i++) {
			probe.selfMoveStr(seq[i]);
		}
		if (probe.ori !== 0) {
			continue;
		}
		var before = inverseState(seq);
		if (!isF2LDone(before) || !edgesOriented(before)) {
			continue;
		}
		var after = new mathlib.CubieCube();
		after.init(before.ca.slice(), before.ea.slice());
		for (var i = 0; i < seq.length; i++) {
			after.selfMoveStr(seq[i]);
		}
		if (!isF2LDone(after) || !edgesOriented(after) || !cornersSolved(after)) {
			continue;
		}
		var score = [ROTATIONS[r].length ? 1 : 0, seq.length, countSlices(seq)];
		if (!best || less(score, best.score)) {
			best = {tokens: seq, score: score};
		}
	}
	return best && best.tokens;
}

function countSlices(ts) {
	var n = 0;
	for (var i = 0; i < ts.length; i++) {
		if (/^[EMS]/.test(ts[i]) || /w/.test(ts[i])) {
			n++;
		}
	}
	return n;
}

function less(a, b) {
	for (var i = 0; i < a.length; i++) {
		if (a[i] != b[i]) {
			return a[i] < b[i];
		}
	}
	return false;
}

var data = JSON.parse(fs.readFileSync(srcFile, 'utf8'));
var out = [];
var missing = [];
var skipped = 0;

for (var ci = 0; ci < data.length; ci++) {
	var entry = data[ci];
	var best = null;
	for (var ai = 0; ai < entry.algs.length; ai++) {
		var ts = tokensOf(entry.algs[ai]);
		if (!validTokens(ts)) {
			skipped++;
			continue;
		}
		var alg = usableAlg(ts);
		if (alg && (!best || less(alg.length, best.length))) {
			best = alg;
		}
	}
	if (!best) {
		missing.push(entry.name);
		continue;
	}
	out.push(entry.name + '|' + best.join(' '));
}

var js = '// COLL algorithms, generated by experiment/gen_coll.js\n' +
	'// Source: speedcubedb.com/a/3x3/COLL.  Every entry solves the last-layer\n' +
	'// corners while keeping the edges oriented and F2L intact, leaving an EPLL.\n' +
	'var COLL_DATA = [\n' +
	out.map(function(s) {
		return '\t' + JSON.stringify(s) + '';
	}).join(',\n') + '\n];\n';

console.log('entries', out.length, 'missing', missing.length, 'skipped', skipped);
if (missing.length) {
	console.log('missing cases:', missing.join(', '));
}
if (write) {
	fs.writeFileSync(outFile, js);
	console.log('written', path.relative(root, outFile));
} else {
	console.log('(dry run, pass --write to update the file)');
}
