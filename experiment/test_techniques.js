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
check('WV also skips OLL', has(wv, 'ollskip'));

var ols = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false},
	{prog: 2, eo: true, co: true, cp: false}]);
check('OLS', has(ols, 'ols'));
check('OLS also skips OLL', has(ols, 'ollskip'));

var ollskip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: true},
	{prog: 2, eo: true, co: true, cp: false}]);
check('OLL skip when the last pair adds no orientation', has(ollskip, 'ollskip'));
check('OLL skip is not OLS', !has(ollskip, 'ols'));

var noOllskip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false},
	{prog: 2, eo: true, co: false}]);
check('no OLL skip when corners remain unoriented', !has(noOllskip, 'ollskip'));

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

var collDip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 2, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true},
	{prog: 2, eo: true, co: true, cp: true},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('COLL is not repeated when PLL dips through F2L', keys(collDip).filter(function(k) {
	return k == 'coll';
}).length == 1);

// ---------------------------------------------------------------- flow
var flow = run(7, [
	{prog: 6, t: 1000},                       // cross
	{prog: 6, t: 1100, eo: true},             // next move within the pause
	{prog: 5, t: 1300},                       // F1
	{prog: 4, t: 1400},                       // F2
	{prog: 3, t: 1500},                       // F3
	{prog: 2, t: 1600},                       // F4
	{prog: 1, t: 1700, eo: true, co: false},  // OLL starts
	{prog: 1, t: 1800, eo: true, co: true},   // OLL done
	{prog: 1, t: 1900, eo: true, co: true, cp: false},
	SOLVED
]);
check('Cross->F1 flow', keys(flow).filter(function(k) {
	return k == 'flowcf';
}).length == 1);
check('pair chains flow', keys(flow).filter(function(k) {
	return k == 'flowff';
}).length == 3);
check('F2L->LL flow', has(flow, 'flowfl'));
check('OLL->PLL flow', has(flow, 'flowop'));

var slowFlow = run(7, [
	{prog: 6, t: 1000},
	{prog: 6, t: 4000},
	{prog: 5, t: 4100},
	{prog: 5, t: 8000}
]);
check('a long pause breaks the flow', !has(slowFlow, 'flowcf') &&
	!has(slowFlow, 'flowff'));

var jumpFlow = run(7, [{prog: 5, t: 1000}, {prog: 5, t: 1100}]);
check('a skipped stage does not invent flow', !has(jumpFlow, 'flowcf'));

// ---------------------------------------------------------------- speed tiers
check('TPS 8 also reaches TPS 6',
	copilot.tpsTiers(8).join(',') == '6,8');
check('TPS 10 reaches every speed tip',
	copilot.tpsTiers(10).join(',') == '6,8,10');
check('TPS 5 reaches no speed tip', copilot.tpsTiers(5).length == 0);

// ---------------------------------------------------------------- So Close
function chokeState(steps) {
	var t = copilot.makeTechniqueTracker();
	t.reset(7);
	steps.forEach(function(s) {
		t.step(s);
	});
	return t.state();
}

var realChoke = chokeState([
	{prog: 6}, {prog: 5}, {prog: 4}, {prog: 3}, {prog: 2},
	{prog: 1, eo: true, co: true, t: 1000},
	{prog: 7, t: 1600},
	{prog: 6, t: 1900},
	{prog: 5, t: 2250},
	{prog: 4, t: 2600},
	{prog: 3, t: 2950},
	{prog: 2, t: 3300},
	{prog: 1, eo: true, co: true, t: 3650}
]);
check('So Close: rebuilding pairs one by one counts', realChoke.choke);

var pllDip = chokeState([
	{prog: 1, eo: true, co: true, t: 1000},
	{prog: 7, t: 1200},
	{prog: 4, t: 1350},
	{prog: 7, t: 1500},
	{prog: 3, t: 1650},
	{prog: 7, t: 1800},
	{prog: 1, eo: true, co: true, t: 1950},
	{prog: 0, t: 2100}
]);
check('So Close: a PLL dip is not a rebuild', !pllDip.choke);

var onePair = chokeState([
	{prog: 1, eo: true, co: true, t: 1000},
	{prog: 7, t: 1200},
	{prog: 3, t: 1500},
	{prog: 2, t: 1800},
	{prog: 1, eo: true, co: true, t: 2100}
]);
check('So Close: a single last pair is not enough', !onePair.choke);

var ollcp = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('OLLCP', has(ollcp, 'ollcp'));
check('OLLCP fires when OLL completes', ollcp.per[5].some(function(e) {
	return e.key == 'ollcp';
}));

var ollcpLate = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true}]);
check('a bare corner-permutation step is not enough for OLLCP',
	!has(ollcpLate, 'ollcp'));

// Real OLLCP/COLL algorithms finish the orientation one or two moves before
// the corner permutation.  The tip is held for one state and only confirmed
// when an EPLL state follows.
var ollcpLag = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('OLLCP when corner permutation lags the orientation', has(ollcpLag, 'ollcp'));
check('lagging OLLCP is reported once', keys(ollcpLag).filter(function(k) {
	return k == 'ollcp';
}).length == 1);
check('lagging OLLCP fires at the EPLL state', ollcpLag.per[7].some(function(e) {
	return e.key == 'ollcp';
}));
check('lagging OLLCP still ends as EPLL', has(ollcpLag, 'epll'));

// The EPLL algorithm usually breaks the F2L on its first move, which hides
// the last layer for a sample or two; the candidate must survive that.
var ollcpHidden = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true},
	{prog: 7},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('OLLCP survives a hidden last-layer sample', has(ollcpHidden, 'ollcp'));
check('hidden OLLCP still ends as EPLL', has(ollcpHidden, 'epll'));

var collLag = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 2, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('COLL when corner permutation lags the orientation', has(collLag, 'coll'));
check('lagging COLL is not OLLCP', !has(collLag, 'ollcp'));

// Corners already oriented after F2L (CO done) means the following algorithm
// is not doing OLL, so it cannot be OLLCP even when it also fixes CP.
var coFirst = run(7, [CROSS_ONLY, P1, P2,
	{prog: 3, eo: false, co: true, cp: false},
	{prog: 2, eo: false, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('CO-complete start is not OLLCP', !has(coFirst, 'ollcp'));
check('CO-complete start is not COLL', !has(coFirst, 'coll'));
check('CO-complete start still ends as EPLL', has(coFirst, 'epll'));

var coFirstLag = run(7, [CROSS_ONLY, P1, P2,
	{prog: 3, eo: false, co: true, cp: false},
	{prog: 2, eo: false, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('CO-complete start with lagging CP is not OLLCP', !has(coFirstLag, 'ollcp'));

// A normal PLL whose corners are only placed on its final move must not be
// mistaken for COLL/OLLCP because of that transient state.
var pllG = run(7, [CROSS_ONLY, P1, P2, {prog: 2, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('a G-perm transient is not OLLCP', !has(pllG, 'ollcp'));
check('a G-perm transient is not COLL', !has(pllG, 'coll'));
check('a G-perm transient is still PLL', has(pllG, 'pll'));
var pllLateSteps = [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false}];
for (var pc = 0; pc < 12; pc++) {
	pllLateSteps.push({prog: 1, eo: true, co: true, cp: false});
}
pllLateSteps.push({prog: 1, eo: true, co: true, cp: true});
check('a normal PLL does not turn into OLLCP', !has(run(7, pllLateSteps), 'ollcp'));

var llll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false}, SOLVED]);
check('1LLL', has(llll, '1lll'));

var pll = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false}, SOLVED]);
check('PLL', has(pll, 'pll'));
check('PLL is not ZBLL', !has(pll, 'zbll'));

var epll = run(7, [CROSS_ONLY, P1, P2, {prog: 2, eo: true, co: true, cp: true}, SOLVED]);
check('EPLL when the corners are already placed', has(epll, 'epll'));
check('EPLL is not reported as PLL', !has(epll, 'pll'));

var ollcpEpll = run(7, [CROSS_ONLY, P1, P2, {prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('EPLL after OLLCP', has(ollcpEpll, 'epll'));
check('OLLCP is not EPLL', has(ollcpEpll, 'ollcp'));

var collEpll = run(7, [CROSS_ONLY, P1, P2, {prog: 2, eo: true, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: true}, SOLVED]);
check('EPLL after COLL', has(collEpll, 'epll'));
check('COLL is not EPLL', has(collEpll, 'coll'));

var llskip = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: true, co: false, cp: false}, SOLVED]);
check('LL skip', has(llskip, 'llskip'));

var normal = run(7, [CROSS_ONLY, P1, P2, {prog: 3, eo: false, co: false, cp: false},
	{prog: 2, eo: false, co: false, cp: false},
	{prog: 1, eo: true, co: true, cp: false}, SOLVED]);
check('normal OLL+PLL has no achievement', noEvents(normal));

// ---------------------------------------------------------------- misc
var multi = run(7, [CROSS_ONLY, P1, {prog: 3}]);
check('Multislotting', has(multi, 'multi'));
var lateMulti = run(7, [CROSS_ONLY, P1, P2, {prog: 3}, {prog: 2},
	{prog: 4}, {prog: 2}]);
check('no multislot after F2L completes', !has(lateMulti, 'multi'));
var lastPair = run(7, [CROSS_ONLY, P1, P2, {prog: 3}, {prog: 4},
	{prog: 3}, {prog: 2}, {prog: 4}, {prog: 2}]);
check('no multislot while inserting the last pair', !has(lastPair, 'multi'));

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
check('Sune: corners placed up to AUF', sune.cp);

var tperm = copilot.techniqueState(faceletOf("R U R' U' R' F R2 U' R' U' R U R' F'"));
check('T perm: EO done', tperm.eo);
check('T perm: corners oriented', tperm.co);
check('T perm: corners unpermuted', !tperm.cp);

var uperm = copilot.techniqueState(faceletOf("R U' R U R U R U' R' U' R2"));
check('U perm: corners placed', uperm.cp);

var upermAuf = copilot.techniqueState(faceletOf("R U' R U R U R U' R' U' R2 U"));
check('U perm with AUF: corners still placed', upermAuf.cp);
check('U perm with AUF: corners still oriented', upermAuf.co);
check('U perm with AUF: edges oriented', upermAuf.eo);

var scrambled = faceletOf("R U F L D B R2 U2 F2");
check('scrambled progress', cubeutil.getProgress(scrambled, 'cf4op') == 7);
check('scrambled has no top cross', !copilot.techniqueState(scrambled).eo);

var basicState = faceletOf("R U R'");
check('basic F2L: joined pair in the U layer', copilot.basicPair(basicState, 0));
check('basic F2L: split pairs are not basic',
	!copilot.basicPair(faceletOf("R U F'"), 0));
check('basic F2L: a solved pair is not a case',
	!copilot.basicPair(faceletOf(''), 0));

// ---------------------------------------------------------------- real algs
// Replay a real last-layer algorithm move by move, the way the app samples
// it, from the state right after F2L.
var UPERM = "R U' R U R U R U' R' U' R2";
var TPERM = "R U R' U' R' F R2 U' R' U' R U R' F'";

function invTok(m) {
	if (m.slice(-1) == "'") {
		return m.slice(0, -1);
	}
	return m.slice(-1) == '2' ? m : m + "'";
}

function invAlg(alg) {
	return alg.trim().split(/\s+/).reverse().map(invTok).join(' ');
}

function replayAlg(setup, alg, tail, llEo) {
	var cc = new mathlib.CubieCube();
	(setup + ' ' + invAlg(alg)).trim().split(/\s+/).forEach(function(m) {
		if (m) {
			cc.selfMoveStr(m);
		}
	});
	var caseState = copilot.techniqueState(cc.toFaceCube());
	var t = copilot.makeTechniqueTracker();
	t.reset(7);
	t.step({prog: 3, eo: llEo, co: false});
	t.step({prog: 2, eo: caseState.eo, co: caseState.co, cp: caseState.cp});
	var keys = [];
	(alg + ' ' + tail).trim().split(/\s+/).forEach(function(m) {
		if (!m) {
			return;
		}
		cc.selfMoveStr(m);
		var facelet = cc.toFaceCube();
		var prog = cubeutil.getProgress(facelet, 'cf4op');
		var sn = {prog: prog};
		if (prog <= 3) {
			var st = copilot.techniqueState(facelet);
			sn.eo = st.eo;
			sn.co = st.co;
			sn.cp = st.cp;
		}
		t.step(sn).forEach(function(e) {
			keys.push(e.key);
		});
	});
	return {keys: keys, st: caseState};
}

// Every PLL algorithm temporarily breaks the F2L while it runs.  None of them
// may be mistaken for a real "So Close" recovery.
var realPll = {n: 0, choke: []};
ctx.PLL_DATA.forEach(function(entry) {
	var ix = entry.indexOf('|');
	var name = entry.slice(0, ix);
	var alg = entry.slice(ix + 1);
	var cc = new mathlib.CubieCube();
	try {
		invAlg(alg).trim().split(/\s+/).forEach(function(m) {
			if (m) {
				cc.selfMoveStr(m);
			}
		});
	} catch (e) {
		return;
	}
	var first = cc.toFaceCube();
	var t = copilot.makeTechniqueTracker();
	t.reset(7);
	t.step({prog: cubeutil.getProgress(first, 'cf4op'), eo: true, co: true,
		t: 1000});
	alg.trim().split(/\s+/).forEach(function(m, idx) {
		cc.selfMoveStr(m);
		var facelet = cc.toFaceCube();
		var prog = cubeutil.getProgress(facelet, 'cf4op');
		var sn = {prog: prog, t: 1150 + idx * 150};
		if (prog <= 3) {
			var st = copilot.techniqueState(facelet);
			sn.eo = st.eo;
			sn.co = st.co;
			sn.cp = st.cp;
		}
		t.step(sn);
	});
	realPll.n++;
	if (t.state().choke) {
		realPll.choke.push(name);
	}
});
check('real PLL algorithms are never So Close (' + realPll.n + ' checked' +
	(realPll.choke.length ? ', false: ' + realPll.choke.join(' ') : '') + ')',
	realPll.n == ctx.PLL_DATA.length && realPll.choke.length == 0);

var realColl = {n: 0, miss: []};
ctx.COLL_DATA.forEach(function(entry) {
	var ix = entry.indexOf('|');
	var r;
	try {
		r = replayAlg(UPERM, entry.slice(ix + 1), UPERM, true);
	} catch (e) {
		return;
	}
	if (r.st.cp || r.st.co) {
		// the corners were already placed or oriented after F2L, so the
		// algorithm did not do COLL
		return;
	}
	realColl.n++;
	if (r.keys.indexOf('coll') < 0) {
		realColl.miss.push(entry.slice(0, ix));
	}
});
check('real COLL algorithms report COLL (' + realColl.n + ' checked' +
	(realColl.miss.length ? ', missed ' + realColl.miss.join(' ') : '') + ')',
	realColl.n > 20 && realColl.miss.length == 0);

// An OLLCP finish: the corners are not oriented or placed after F2L and one
// algorithm leaves an EPLL.  Every ALLL algorithm plus a U perm models one.
var realOllcp = {n: 0, miss: []};
ctx.ALLL_DATA.forEach(function(entry) {
	var ix = entry.indexOf('|');
	var r;
	try {
		r = replayAlg(UPERM, entry.slice(ix + 1), UPERM, false);
	} catch (e) {
		return;
	}
	if (r.st.eo || r.st.co || r.st.cp) {
		// edges already oriented or corners already oriented/placed:
		// not an OLLCP start
		return;
	}
	realOllcp.n++;
	if (r.keys.indexOf('ollcp') < 0) {
		realOllcp.miss.push(entry.slice(0, ix));
	}
});
check('real OLLCP finishes report OLLCP (' + realOllcp.n + ' checked' +
	(realOllcp.miss.length ? ', missed ' + realOllcp.miss.slice(0, 8).join(' ') : '') + ')',
	realOllcp.n > 3000 && realOllcp.miss.length == 0);

var realNormal = {n: 0, bad: []};
ctx.OLL_DATA.forEach(function(entry) {
	var ix = entry.indexOf('|');
	var r;
	try {
		r = replayAlg(TPERM, entry.slice(ix + 1), TPERM, false);
	} catch (e) {
		return;
	}
	realNormal.n++;
	if (r.keys.indexOf('coll') >= 0 || r.keys.indexOf('ollcp') >= 0) {
		realNormal.bad.push(entry.slice(0, ix));
	}
});
check('a real OLL+PLL solve never reports COLL/OLLCP', realNormal.bad.length == 0);

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
