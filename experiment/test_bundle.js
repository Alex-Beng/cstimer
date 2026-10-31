"use strict";

// Smoke test: load the worker bundle (moduleSrc) in a node context and check
// that the copilot entry points exist and answer a hint.
// Usage: node experiment/test_bundle.js

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var mk = fs.readFileSync(path.join(root, 'Makefile'), 'utf8');

function srcList(name) {
	var start = mk.indexOf(name + ' = ');
	if (start < 0) {
		throw new Error('target not found: ' + name);
	}
	var end = mk.indexOf('\n\n', start);
	var block = mk.slice(start, end < 0 ? mk.length : end);
	return [...block.matchAll(/^\s*([\w\/.]+\.js)\s*\\?\s*$/gm)].map(function(m) {
		return m[1];
	});
}

var files = srcList('moduleSrc');
console.log('moduleSrc files:', files.length);

var ctx = {
	console: console,
	process: {env: {}},
	module: {exports: {}},
	Date: Date,
	Math: Math,
	JSON: JSON,
	setTimeout: setTimeout,
	setInterval: setInterval,
	clearTimeout: clearTimeout,
	self: {},
	Uint8Array: Uint8Array,
	Int8Array: Int8Array,
	Float64Array: Float64Array,
	Promise: Promise,
	navigator: {}
};
ctx.window = ctx;
ctx.localStorage = {};
vm.createContext(ctx);
vm.runInContext('var $={isArray:Array.isArray,noop:function(){},now:function(){return Date.now();},urlParam:function(){return "";}};', ctx);

var failures = 0;
files.forEach(function(f) {
	var code = fs.readFileSync(path.join(root, 'src/js', f), 'utf8');
	if (f == 'lib/utillib.js') {
		// ISCSTIMER is provided by the Closure build defines
		code = code.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;');
	}
	try {
		vm.runInContext(code, ctx, {filename: f});
	} catch (e) {
		// browser-only modules may legitimately fail here; report once
		failures++;
		console.log('LOAD FAIL', f, e.message.split('\n')[0]);
	}
});

console.log('loaded with', failures, 'failures');
console.log('cubeCopilot:', typeof ctx.cubeCopilot);
console.log('scrambleGate:', typeof ctx.scrambleGate);

if (ctx.cubeCopilot) {
	var m = ctx.mathlib;
	var cc = new m.CubieCube();
	for (var i = 0; i < 3; i++) {
		cc.selfMoveStr('R');
	}
	var stage = ctx.cubeCopilot.getStage(cc.toFaceCube());
	console.log('stage of R3:', stage);
	var t = Date.now();
	var stats = ctx.cubeCopilot.buildAlllLut();
	console.log('alll build ms:', Date.now() - t, JSON.stringify(stats));
}

process.exitCode = ctx.cubeCopilot && ctx.scrambleGate ? 0 : 1;
