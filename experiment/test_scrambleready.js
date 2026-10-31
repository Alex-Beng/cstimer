"use strict";

// Unit test for lib/scrambleready.js, the fix for "next solve uses the previous
// scramble" in the virtual cube.
// Usage: node experiment/test_scrambleready.js

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var ctx = {console: console, process: {env: {}}, module: {exports: {}}, self: {}};
vm.createContext(ctx);
vm.runInContext('var $={};', ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/isaac.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
	.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/scrambleready.js'), 'utf8'), ctx);

var gate = ctx.scrambleGate;
var failures = 0;

function check(name, actual, expected) {
	if (actual !== expected) {
		failures++;
		console.log('FAIL ' + name + ': got ' + actual + ', want ' + expected);
	}
}

// idle: nothing in flight, the text on screen is usable
check('idle ready', gate.isReady(), true);

// a new generation starts: the old text must not be applied
gate.beginGenerating();
check('generating not ready', gate.isReady(), false);

// repeated begin (rapid next/next) stays not-ready
gate.beginGenerating();
check('still generating', gate.isReady(), false);

// the fresh text arrives
gate.publish();
check('published ready', gate.isReady(), true);

// token must advance across generations so stale callbacks can be dropped
var t1 = gate.token();
gate.beginGenerating();
gate.publish();
var t2 = gate.token();
check('token advances', t2 > t1, true);

// explicit reset also invalidates
gate.beginGenerating();
gate.reset();
check('reset ready', gate.isReady(), true);
var t3 = gate.token();
check('reset advances token', t3 > t2, true);

// an empty placeholder (e.g. mode switched off) also counts as published
gate.beginGenerating();
gate.publish();
check('placeholder ready', gate.isReady(), true);

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
