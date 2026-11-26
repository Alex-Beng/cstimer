"use strict";

// Verify the advanced-technique achievement state machine:
//   - XC / XXC / XXXC / XXXXC fire when the cross finishes with pairs solved
//   - ZBLS, WV/SV and OLL skip fire when the last pair completes the top layer
//   - ZBLL / COLL / PLL / OLLCP / 1LLL / LL skip fire at the end of the solve
//   - a normal OLL + PLL solve produces no achievement
// Usage: node experiment/test_techniques.js

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');

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
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/cubeutil.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/ollpll.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/zbll.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/coll.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/alll.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/copilot.js'), 'utf8'), ctx);

var mathlib = ctx.mathlib;
var cubeutil = ctx.cubeutil;
var copilot = ctx.cubeCopilot;

var failures = 0;
function check(name, cond) {
	if (!cond) {
		failures++;
		console.log('FAIL ' + name);
	}
}

function run(seed, steps) {
	var t = copilot.makeTechniqueTracker();
	t.reset(seed);
	var all = [];
	var per = [];
	steps.forEach(function(s) {
		var ev = t.step(s);
		per.push(ev);
		all = all.concat(ev);
	});
	return {all: all, per: per};
}

function keys(res) {
	return res.all.map(function(e) {
		return e.key;
	});
}

function has(res, key) {
	return keys(res).indexOf(key) != -1;
}

function noEvents(res) {
	return res.all.length == 0;
}

// 7 = nothing, 6 = cross, 5..2 = pairs, 1 = OLL, 0 = solved
var CROSS_ONLY = {prog: 6};
var P1 = {prog: 5};
var P2 = {prog: 4};
var P3 = {prog: 3};
var F2L = {prog: 2};
var OLL = {prog: 1};
var SOLVED = {prog: 0};

// ---------------------------------------------------------------- extended cross
check('XC', has(run(7, [{prog: 5}]), 'xc'));
check('XXC', has(run(7, [{prog: 4}]), 'xxc'));
check('XXXC', has(run(7, [{prog: 3}]), 'xxxc'));
check('XXXXC', has(run(7, [{prog: 2}]), 'xxxxc'));
check('cross only is not XC', !has(run(7, [CROSS_ONLY]), 'xc'));
check('seeded cross does not re-fire XC', noEvents(run(6, [P1])));
check('unknown baseline does not invent an extended cross', noEvents(run(-1, [SOLVED])));

// ---------------------------------------------------------------- last slot
var zbls = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false},
	{prog: 2, eo: true, co: false, cp: false}]);
check('ZBLS', has(zbls, 'zbls'));
check('ZBLS is not WV', !has(zbls, 'wv'));

var wv = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false},
	{prog: 2, eo: true, co: true, cp: false}]);
check('WV/SV', has(wv, 'wv'));
check('WV is not ZBLS', !has(wv, 'zbls'));

var ollskip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false},
	{prog: 2, eo: true, co: true, cp: false}]);
check('OLL skip', has(ollskip, 'ollskip'));

var eoFirst = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false},
	{prog: 2, eo: true, co: false}]);
check('EO done before the last pair is not ZBLS', !has(eoFirst, 'zbls'));

// ---------------------------------------------------------------- last layer
var zbll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 2, eo: true, co: false, cp: false}, SOLVED]);
check('ZBLL', has(zbll, 'zbll'));

var coll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 2, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('COLL', has(coll, 'coll'));
check('COLL fires when OLL completes', coll.per[5].some(function(e) {
	return e.key == 'coll';
}));
check('COLL is not repeated at PLL', !coll.per[6].some(function(e) {
	return e.key == 'coll';
}));

var ollcp = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('OLLCP', has(ollcp, 'ollcp'));
check('OLLCP fires when OLL completes', ollcp.per[5].some(function(e) {
	return e.key == 'ollcp';
}));

var llll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false}, SOLVED]);
check('1LLL', has(llll, '1lll'));

var pll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false}, SOLVED]);
check('PLL', has(pll, 'pll'));
check('PLL is not ZBLL', !has(pll, 'zbll'));

var llskip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false}, SOLVED]);
check('LL skip', has(llskip, 'llskip'));

var normal = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false}, SOLVED]);
check('normal OLL+PLL has no achievement', noEvents(normal));

// ---------------------------------------------------------------- misc
var multi = run(7, [CROSS_ONLY, P1, {prog: 3}]);
check('Multislotting', has(multi, 'multi'));

var regress = run(7, [P1, {prog: 7}, P1]);
check('a broken cross does not re-fire XC', keys(regress).filter(function(k) {
	return k == 'xc';
}).length == 1);

// ---------------------------------------------------------------- real facelets
function faceletOf(seq) {
	var cc = new mathlib.CubieCube();
	seq.split(' ').forEach(function(m) {
		if (m) {
			cc.selfMoveStr(m);
		}
	});
	cc.ori = 0;
	return cc.toFaceCube();
}

var sune = copilot.techniqueState(faceletOf("R U R' U R U2 R'"));
check('Sune: EO done', sune.eo);
check('Sune: corners unoriented', !sune.co);
check('Sune: corners unpermuted', !sune.cp);

var tperm = copilot.techniqueState(faceletOf("R U R' U' R' F R2 U' R' U' R U R' F'"));
check('T perm: EO done', tperm.eo);
check('T perm: corners oriented', tperm.co);
check('T perm: corners unpermuted', !tperm.cp);

var uperm = copilot.techniqueState(faceletOf("R U' R U R U R U' R' U' R2"));
check('U perm: corners placed', uperm.cp);

var scrambled = faceletOf("R U F L D B R2 U2 F2");
check('scrambled progress', cubeutil.getProgress(scrambled, 'cf4op') == 7);
check('scrambled has no top cross', !copilot.techniqueState(scrambled).eo);

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
