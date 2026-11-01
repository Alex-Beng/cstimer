"use strict";

// Generate src/js/lib/alll.js from ../MOYU/solve/ALLL.py.
//
// Extraction rules (matching MOYU's own `algs_str.splitlines()[1:]`):
//   * each group is a triple-quoted block
//   * every non-empty line inside a block is ONE complete algorithm; the
//     surrounding double quotes only group alternatives and are stripped
//   * annotations (·, ↑, ↓, =, wyh) are noise
//
// Moves are kept in csTimer notation (Rw / M / x ...) so that the runtime can
// feed them straight to CubieCube.selfMoveStr, which tracks the net whole-cube
// orientation.  Expanding wide/slice moves by hand would corrupt that frame.
//
// Usage: node experiment/gen_alll.js [moyu/solve/ALLL.py] [--write]

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var srcFile = process.argv[2] && process.argv[2].charAt(0) != '-' ? process.argv[2] :
	path.resolve(root, '..', 'MOYU', 'solve', 'ALLL.py');

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

// One alg per line.  Quotes are group delimiters, not string continuations.
function extractGroups(py) {
	var groups = [];
	var re = /^(\w+)\s*=\s*"""([\s\S]*?)"""/gm;
	var m;
	while ((m = re.exec(py)) != null) {
		var lines = m[2].split('\n').map(function(raw) {
			return raw.replace(/[·↑↓"]/g, ' ').replace(/=/g, ' ')
				.replace(/\bwyh\b/g, ' ').trim();
		}).filter(function(line) {
			return line.length > 0;
		});
		groups.push([m[1], lines]);
	}
	return groups;
}

// "(seq)2" -> "seq seq"
function expandBrackets(str) {
	var re = /\(([^()]*)\)\s*(2|3|4)?'?/;
	var guard = 0;
	var m;
	while ((m = re.exec(str)) != null && guard++ < 100) {
		var n = ~~m[2] || 1;
		var rep = [];
		for (var i = 0; i < n; i++) {
			rep.push(m[1]);
		}
		str = str.slice(0, m.index) + ' ' + rep.join(' ') + ' ' + str.slice(m.index + m[0].length);
	}
	return str;
}

/**
 * Parse an algorithm into csTimer tokens:
 *   - lowercase wide turns (r, u, ...) become Rw, Uw, ...
 *   - "X2'" is the reverse half turn in the MOYU data but csTimer treats a
 *     half turn as self-inverse, so it is normalised to "X2"
 *   - slice moves (M/E/S) and rotations (x/y/z) are kept as-is
 */
function parseAlg(alg) {
	var raw = expandBrackets(alg).replace(/[()\[\]]/g, ' ');
	var ret = [];
	var i = 0;
	while (i < raw.length) {
		var ch = raw.charAt(i);
		if (/\s/.test(ch)) {
			i++;
			continue;
		}
		if ('URFDLBurfdlbMESxyz'.indexOf(ch) == -1) {
			i++;
			continue;
		}
		var base = ch;
		i++;
		var wide = '';
		if (raw.charAt(i) == 'w') {
			wide = 'w';
			i++;
		}
		var mod = '';
		if (raw.charAt(i) == '2') {
			mod = '2';
			i++;
			if (raw.charAt(i) == "'") {
				i++;
			}
		} else if (raw.charAt(i) == "'") {
			mod = "'";
			i++;
		}
		if ('urfdlb'.indexOf(base) != -1) {
			base = base.toUpperCase();
			wide = 'w';
		}
		ret.push(base + wide + mod);
	}
	if (!ret.length) {
		throw new Error('empty alg');
	}
	return ret;
}

function invertTokens(tokens) {
	var ret = [];
	for (var i = tokens.length - 1; i >= 0; i--) {
		var m = /^([URFDLB]w?|[EMSxyz])(2|'|2')?$/.exec(tokens[i]);
		if (!m) {
			ret.push(tokens[i]);
			continue;
		}
		var mod = m[2] || '';
		if (mod == '2' || mod == "2'") {
			ret.push(m[1] + '2');
		} else if (mod == "'") {
			ret.push(m[1]);
		} else {
			ret.push(m[1] + "'");
		}
	}
	return ret;
}

// ---------------------------------------------------------------- inversion

var ROT_AXIS = [3, 15, 17, 1, 11, 23];

function powOf(mod) {
	if (mod == "'") {
		return 3;
	}
	if (mod == '2' || mod == "2'") {
		return 2;
	}
	return 1;
}

/**
 * Face-move indices actually applied by one token, in the given frame.
 * Mirrors mathlib.CubieCube.prototype.selfMoveStr: wide turns apply the
 * opposite face, slice turns apply two faces, rotations apply none.
 */
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
		var m1 = saxis * 3 + (4 - spow) % 4 - 1;
		var m2 = (saxis + 3) % 6 * 3 + spow % 4 - 1;
		return [
			mathlib.CubieCube.rotMulM[ori][m1],
			mathlib.CubieCube.rotMulM[ori][m2]
		];
	}
	return []; // whole-cube rotation: no cubie move
}

function inverseMove(idx) {
	return ~~(idx / 3) * 3 + (2 - idx % 3);
}

/**
 * Inverse of a token sequence as a *function*.
 *
 * csTimer's `ori` is part of the state: a wide/slice token changes the frame
 * used by the moves that follow, so simply reversing the tokens with primed
 * moves is NOT the inverse.  The frame only depends on the token sequence, so
 * we record the frame before every token, then undo the applied face moves in
 * reverse order.
 */
function inverseState(tokens) {
	var probe = new mathlib.CubieCube();
	var oriBefore = [];
	for (var i = 0; i < tokens.length; i++) {
		oriBefore.push(probe.ori);
		probe.selfMoveStr(tokens[i]);
	}
	var cc = new mathlib.CubieCube();
	for (var i = tokens.length - 1; i >= 0; i--) {
		var applied = movesAppliedBy(tokens[i], oriBefore[i]);
		for (var k = applied.length - 1; k >= 0; k--) {
			var inv = inverseMove(applied[k]);
			var tmp = new mathlib.CubieCube();
			mathlib.CubieCube.CubeMult(cc, mathlib.CubieCube.moveCube[inv], tmp);
			cc.init(tmp.ca, tmp.ea);
		}
	}
	cc.ori = 0;
	return cc;
}

function applyTokens(cc, tokens) {
	for (var i = 0; i < tokens.length; i++) {
		cc.selfMoveStr(tokens[i]);
	}
	return cc;
}

// A legal 1LLL recognition state keeps the whole F2L intact: the D face and the
// two lower rows of every side face are untouched.
function f2lIntact(cc) {
	var f = cc.toFaceCube();
	for (var k = 27; k < 36; k++) {
		if (f.charAt(k) != mathlib.SOLVED_FACELET.charAt(k)) {
			return false;
		}
	}
	var starts = [9, 18, 36, 45];
	for (var s = 0; s < starts.length; s++) {
		for (var r = 3; r < 9; r++) {
			if (f.charAt(starts[s] + r) != mathlib.SOLVED_FACELET.charAt(starts[s] + r)) {
				return false;
			}
		}
	}
	return true;
}

/**
 * Recognition state: the position which the alg solves, i.e. the function
 * inverse of the token sequence applied to a solved cube.  See inverseState()
 * for why reversing the tokens is not enough.
 */
function recognitionState(tokens) {
	return inverseState(tokens);
}

var groups = extractGroups(fs.readFileSync(srcFile, 'utf8'));
var out = [];
var bad = [];
var brokenSolve = [];
var total = 0;

groups.forEach(function(group) {
	group[1].forEach(function(alg, i) {
		total++;
		var name = group[0] + '_' + (i + 1);
		try {
			var tokens = parseAlg(alg);
			var state = recognitionState(tokens);
			if (!f2lIntact(state)) {
				throw new Error('breaks F2L');
			}
			// end-to-end check: the alg must solve its recognition state
			var replay = new mathlib.CubieCube();
			replay.init(state.ca.slice(), state.ea.slice());
			applyTokens(replay, tokens);
			if (!replay.isEqual(mathlib.CubieCube.SOLVED)) {
				throw new Error('alg does not solve its own state');
			}
			out.push(name + '|' + tokens.join(' '));
		} catch (e) {
			bad.push(name + ': ' + e.message + ' :: ' + alg);
		}
	});
});

console.log('groups', groups.length, 'algs', total, 'valid', out.length, 'failed', bad.length);
if (bad.length) {
	console.log(bad.slice(0, 15).join('\n'));
}

if (process.argv.indexOf('--write') != -1) {
	var js = '// ALL (1LLL) algorithms, data imported from MOYU/solve/ALLL.py\n' +
		'// entry: "NAME|m1 m2 ...", moves use csTimer notation (R, U\', Rw, M, x, ...)\n' +
		'// the alg solves the state reached by applying its inverse to a solved cube;\n' +
		'// recognition LUT (with U/U2/U\' pre- and post-AUF) is built at runtime\n' +
		'//\n' +
		'// ' + (total - out.length) + ' entries were skipped during import: they either\n' +
		'// break the F2L or contain whole-cube rotations that cannot be expressed in\n' +
		'// csTimer\'s fixed cubie indexing (see experiment/gen_alll.js).\n' +
		'var ALLL_DATA = [\n' +
		out.map(function(s) {
			return '\t"' + s + '"';
		}).join(',\n') +
		'\n];\n';
	fs.writeFileSync(path.join(root, 'src/js/lib/alll.js'), js);
	console.log('written src/js/lib/alll.js entries', out.length, 'skipped', total - out.length);
}
