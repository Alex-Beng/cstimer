"use strict";

// Generate src/js/lib/alll.js from ../MOYU/solve/ALLL.py and validate every
// algorithm with csTimer's own CubieCube convention.
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
	var util = fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
		.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;');
	vm.runInContext(util, ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/mathlib.js'), 'utf8'), ctx);
	return ctx.mathlib;
}

// One alg per line. An alg may be wrapped in double quotes across several
// lines; apostrophes are move modifiers and must be preserved.
function extractGroups(py) {
	var groups = [];
	var re = /^(\w+)\s*=\s*"""([\s\S]*?)"""/gm;
	var m;
	while ((m = re.exec(py)) != null) {
		var lines = [];
		var buf = null;
		m[2].split('\n').forEach(function(raw) {
			var line = raw.replace(/[·↑↓]/g, ' ').trim();
			if (buf == null && !line) {
				return;
			}
			if (buf != null) {
				buf += ' ' + line;
				if (/"\s*$/.test(line)) {
					lines.push(buf.replace(/"\s*$/, ''));
					buf = null;
				}
				return;
			}
			if (/^"/.test(line)) {
				if (/"\s*$/.test(line) && line.length > 1) {
					lines.push(line.replace(/^"/, '').replace(/"\s*$/, ''));
				} else {
					buf = line.slice(1);
				}
				return;
			}
			lines.push(line);
		});
		groups.push([m[1], lines.filter(function(s) {
			return /\S/.test(s);
		})]);
	}
	return groups;
}

// (seq)2 -> seq seq ; (seq)3 -> seq seq seq
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

var TOKEN_RE = /^([URFDLBurfdlbMESxyz])(w?)(2|'|2')?$/;

// csTimer's selfMoveStr accepts Uw/Rw/... but not the lowercase wide form.
function toCstimerToken(tok) {
	var m = TOKEN_RE.exec(tok);
	if (!m) {
		return null;
	}
	var base = m[1];
	var wide = m[2];
	var mod = m[3] || '';
	if (mod == "2'") {
		mod = '2';
	}
	if ('urfdlb'.indexOf(base) != -1) {
		base = base.toUpperCase();
		wide = 'w';
	}
	return base + wide + mod;
}

function invertToken(tok) {
	var m = TOKEN_RE.exec(tok);
	if (!m) {
		throw new Error('cannot invert token ' + tok);
	}
	var base = m[1] + m[2];
	var mod = m[3] || '';
	if (mod == "2'" || mod == '2') {
		return base + '2';
	}
	return base + (mod == "'" ? '' : "'");
}

function parseAlg(alg) {
	var raw = expandBrackets(alg).replace(/[()\[\]"]/g, ' ').replace(/[·↑↓]/g, ' ').trim();
	// drop annotations such as [oh,key], (wyh), stray "=" used as a separator
	raw = raw.replace(/\[[^\]]*\]/g, ' ').replace(/\bwyh\b/g, ' ').replace(/=/g, ' ');
	var ret = [];
	var i = 0;
	while (i < raw.length) {
		var ch = raw.charAt(i);
		if (/\s/.test(ch)) {
			i++;
			continue;
		}
		if ('URFDLBurfdlbMESxyz'.indexOf(ch) == -1) {
			throw new Error('unparsable char "' + ch + '"');
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
	if (ret.length == 0) {
		throw new Error('empty alg');
	}
	return ret;
}

function applyTokens(tokens) {
	var cube = new mathlib.CubieCube();
	for (var i = 0; i < tokens.length; i++) {
		cube.selfMoveStr(tokens[i]);
	}
	return cube;
}

function invertTokens(tokens) {
	var ret = [];
	for (var i = tokens.length - 1; i >= 0; i--) {
		ret.push(invertToken(tokens[i]));
	}
	return ret;
}

// The table stores the state which the alg solves: apply the inverse alg to a
// solved cube. A legal 1LLL state has a complete F2L, so the D face and the
// two lower rows of every side face must be untouched; the U layer (U face and
// the top row of each side face) may show any colors.
function checkLastLayerState(cube) {
	var facelet = cube.toFaceCube();
	if (facelet == mathlib.SOLVED_FACELET) {
		return;
	}
	// D face (27..35) and rows 1..2 of R/F/L/B
	var checked = [];
	for (var k = 27; k < 36; k++) {
		checked.push(k);
	}
	[9, 18, 36, 45].forEach(function(start) {
		for (var r = 3; r < 9; r++) {
			checked.push(start + r);
		}
	});
	for (var i = 0; i < checked.length; i++) {
		var k = checked[i];
		if (facelet.charAt(k) != mathlib.SOLVED_FACELET.charAt(k)) {
			throw new Error('breaks F2L at sticker ' + k + ' (' + facelet.charAt(k) + ')');
		}
	}
}

var mathlib = loadMathlib();
var groups = extractGroups(fs.readFileSync(srcFile, 'utf8'));

var AUF = ['', 'U', 'U2', "U'"];
var out = [];
var bad = [];
var byKey = {};
var total = 0;
var valid = 0;

function stateKey(cube) {
	return cube.ca.join(',') + '|' + cube.ea.join(',');
}

groups.forEach(function(group) {
	group[1].forEach(function(alg, i) {
		total++;
		var name = group[0] + '_' + (i + 1);
		var tokens;
		try {
			tokens = parseAlg(alg);
			var base = applyTokens(invertTokens(tokens));
			checkLastLayerState(base);
			valid++;
		} catch (e) {
			bad.push(name + ': ' + e.message);
			return;
		}
		out.push({name: name, alg: tokens.join(' '), tokens: tokens});
		// coverage check: expand pre/post AUF and count distinct states
		for (var pre = 0; pre < 4; pre++) {
			for (var post = 0; post < 4; post++) {
				var seq = [];
				if (AUF[pre]) {
					seq.push(AUF[pre]);
				}
				seq = seq.concat(tokens);
				if (AUF[post]) {
					seq.push(AUF[post]);
				}
				byKey[stateKey(applyTokens(invertTokens(seq)))] = 1;
			}
		}
	});
});

var LL_STATES = 62208; // 4!*3^3 * 4!*2^3 / 2 (parity)
var covered = Object.keys(byKey).length;
console.log('groups', groups.length, 'algs', total, 'valid algs', valid, 'failed', bad.length);
console.log('covered states', covered, '/', LL_STATES, '(' + (covered * 100 / LL_STATES).toFixed(2) + '%)');
if (bad.length) {
	console.log('skipped entries:');
	console.log(bad.slice(0, 40).join('\n'));
}
if (process.argv.indexOf('--write') != -1) {
	var js = '// ALL (1LLL) algorithms, data imported from MOYU/solve/ALLL.py\n' +
		'// entry: "NAME|m1 m2 ...", moves use csTimer notation (R, U\', Rw, M, x, ...)\n' +
		'// the alg solves the state reached by applying its inverse to a solved cube;\n' +
		'// the recognition LUT (with U/U2/U\' pre- and post-AUF) is built at runtime\n' +
		'var ALLL_DATA = [\n' +
		out.map(function(s) {
			return '\t"' + s.name + '|' + s.alg + '"';
		}).join(',\n') +
		'\n];\n';
	fs.writeFileSync(path.join(root, 'src/js/lib/alll.js'), js);
	console.log('written src/js/lib/alll.js, base entries', out.length);
}
