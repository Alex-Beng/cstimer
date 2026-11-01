"use strict";

// Generate src/js/lib/ollpll.js from ../MOYU/solve/cfop_solver.py.
//
// The two dicts OLL_ALGS (57 cases) and PLL_ALGS (21 cases) are the standard
// two-look last-layer algorithm sets.  Usage:
//   node experiment/gen_ollpll.js [cfop_solver.py] [--write]

var fs = require('fs');
var path = require('path');

var root = path.resolve(__dirname, '..');
var srcFile = process.argv[2] && process.argv[2].charAt(0) != '-' ? process.argv[2] :
	path.resolve(root, '..', 'MOYU', 'solve', 'cfop_solver.py');
var outFile = path.resolve(root, 'src', 'js', 'lib', 'ollpll.js');
var write = process.argv.indexOf('--write') >= 0;

function expandBrackets(str) {
	var re = /\(([^()]*)\)\s*(2|3|4)?'?/;
	var guard = 0;
	var m;
	while ((m = re.exec(str)) != null && guard++ < 100) {
		var n = ~~m[2] || 1;
		var rep = [];
		for (var i = 0; i < n; i++) rep.push(m[1]);
		str = str.slice(0, m.index) + ' ' + rep.join(' ') + ' ' + str.slice(m.index + m[0].length);
	}
	return str;
}

function parseAlg(alg) {
	var raw = expandBrackets(alg).replace(/[()\[\]]/g, ' ');
	var ret = [];
	var i = 0;
	while (i < raw.length) {
		var ch = raw.charAt(i);
		if (/\s/.test(ch)) { i++; continue; }
		if ('URFDLBurfdlbMESxyz'.indexOf(ch) == -1) { i++; continue; }
		var base = ch;
		i++;
		var wide = '';
		if (raw.charAt(i) == 'w') { wide = 'w'; i++; }
		var mod = '';
		if (raw.charAt(i) == '2') {
			mod = '2';
			i++;
			if (raw.charAt(i) == "'") i++;
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
	return ret;
}

function extractDict(py, varname) {
	var re = new RegExp('^' + varname + '\\s*=\\s*\\{([\\s\\S]*?)^\\}', 'm');
	var m = re.exec(py);
	if (!m) throw new Error('cannot find ' + varname);
	var body = m[1];
	var out = [];
	var lineRe = /^\s*(?:(\d+)|'([^']+)'|"([^"]+)")\s*:\s*(['"])(.*?)\4\s*,?\s*(?:#.*)?$/gm;
	var lm;
	while ((lm = lineRe.exec(body)) != null) {
		var name = lm[1] || lm[2] || lm[3];
		out.push([name, lm[5]]);
	}
	return out;
}

var py = fs.readFileSync(srcFile, 'utf8');
var oll = extractDict(py, 'OLL_ALGS');
var pll = extractDict(py, 'PLL_ALGS');
if (oll.length != 57 || pll.length != 21) {
	throw new Error('unexpected case counts: OLL=' + oll.length + ' PLL=' + pll.length);
}

function toEntries(list) {
	return list.map(function(e) {
		var tokens = parseAlg(e[1]);
		if (!tokens.length) throw new Error('empty alg for ' + e[0]);
		return e[0] + '|' + tokens.join(' ');
	});
}

var lines = [];
lines.push('// Standard two-look last layer (OLL then PLL), data imported from');
lines.push('// MOYU/solve/cfop_solver.py.  entry: "NAME|m1 m2 ..." in csTimer notation.');
lines.push('// ' + oll.length + ' OLL cases and ' + pll.length + ' PLL cases.');
lines.push('');
lines.push('var OLL_DATA = [');
lines.push(toEntries(oll).map(function(e) { return '\t"' + e + '",'; }).join('\n'));
lines.push('];');
lines.push('');
lines.push('var PLL_DATA = [');
lines.push(toEntries(pll).map(function(e) { return '\t"' + e + '",'; }).join('\n'));
lines.push('];');
lines.push('');
var content = lines.join('\n');

if (!write) {
	console.log(content);
} else {
	fs.writeFileSync(outFile, content);
	console.log('wrote ' + outFile + ' (' + oll.length + ' OLL, ' + pll.length + ' PLL)');
}
