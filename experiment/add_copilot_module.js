"use strict";

// Add the "copilot" module to MODULE_NAMES in every language file so the
// Options panel renders its settings group (kernel.js iterates MODULE_NAMES).
var fs = require('fs');
var path = require('path');

var dir = path.resolve(__dirname, '..', 'src', 'lang');
var changed = 0;

fs.readdirSync(dir).forEach(function(f) {
	if (!/\.js$/.test(f)) return;
	var p = path.join(dir, f);
	var s = fs.readFileSync(p, 'utf8');
	var i = s.indexOf('MODULE_NAMES');
	if (i < 0) return;
	if (/"copilot"\s*:/.test(s)) return; // already present
	var open = s.indexOf('{', i);
	var close = s.indexOf('};', open);
	if (open < 0 || close < 0) {
		console.log('SKIP (unparsable)', f);
		return;
	}
	var head = s.slice(0, close).replace(/\s*$/, '');
	var tail = s.slice(close);
	s = head + ',\n\t"copilot": \'Copilot\'\n' + tail;
	fs.writeFileSync(p, s);
	changed++;
});

console.log('language files updated:', changed);
