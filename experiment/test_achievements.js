"use strict";

// Verify the cross-solve achievement state machine:
//   - streaks count consecutive qualifying solves and reset on a miss
//   - same-solve combos (ZBLS + ZBLL, >=3 techniques)
//   - the codex tracks every technique kind
//   - serialized state survives a reload
// Usage: node experiment/test_achievements.js

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

var copilot = ctx.cubeCopilot;
var failures = 0;

function check(name, cond) {
	if (!cond) {
		failures++;
		console.log('FAIL ' + name);
	}
}

function tracker() {
	var t = copilot.makeAchievementTracker();
	// Most assertions below test one rule in isolation.  Pretend the very
	// first solve is already history so the staged "first solve" card does
	// not add an unrelated event to every fresh tracker.
	t.load({solves: 1, unlocked: {solves1: 1}});
	return t;
}

function step(t, keys, stats) {
	return t.step(keys, stats).map(function(ev) {
		return ev.key;
	});
}

function has(list, key) {
	return list.indexOf(key) != -1;
}

// ---------------------------------------------------------------- streaks
var t = tracker();
check('no achievement after one ZBLL', step(t, ['zbll']).length == 0);
check('no achievement after two ZBLL', step(t, ['zbll']).length == 0);
var ev = step(t, ['zbll']);
check('ZB 狂人 after three ZBLL in a row', has(ev, 'zb3'));
check('手感火热 after three technique solves', has(ev, 'hot3'));
check('ZB 狂人 is not repeated', !has(step(t, ['zbll']), 'zb3'));

t = tracker();
step(t, ['zbll']);
step(t, []);
check('a plain solve resets the ZBLL streak', !has(step(t, ['zbll']), 'zb3'));
check('streak restarts from one after a reset', !has(step(t, ['zbll']), 'zb3'));
check('streak reaches three after a reset', has(step(t, ['zbll']), 'zb3'));

/**
 * A plain solve is one with no technique at all.  The counter above was
 * reset by it, so three fresh ZBLL solves are needed again.
 */
t = tracker();
step(t, ['zbll']);
step(t, ['zbll']);
check('streak is not inherited', !has(step(t, ['coll']), 'zb3'));

// ---------------------------------------------------------- family streaks
t = tracker();
step(t, ['coll']);
step(t, ['coll']);
check('COLL 收割机', has(step(t, ['coll']), 'coll3'));

t = tracker();
step(t, ['zbls']);
step(t, ['ols']);
check('插槽艺术家', has(step(t, ['wv']), 'slot3'));

t = tracker();
step(t, ['xc']);
step(t, ['xxc']);
check('十字军', has(step(t, ['xxxc']), 'xc3'));

// ------------------------------------------------------------------ speed
function stepStats(t, keys, stats) {
	return t.step(keys, stats).map(function(ev) {
		return ev.key;
	});
}

t = tracker();
check('Lightspeed at 10 TPS', has(stepStats(t, [], {tps: 10}), 'speed'));
check('Lightspeed is not repeated', !has(stepStats(t, [], {tps: 12}), 'speed'));

t = tracker();
stepStats(t, [], {tps: 8});
stepStats(t, [], {tps: 8});
check('手速狂人 after three 8+ TPS solves', has(stepStats(t, [], {tps: 9}), 'tps3'));

t = tracker();
stepStats(t, [], {tps: 8});
stepStats(t, [], {tps: 8});
stepStats(t, [], {tps: 5});
check('a slow solve resets the TPS streak', !has(stepStats(t, [], {tps: 8}), 'tps3'));
check('TPS streak restarts from one', !has(stepStats(t, [], {tps: 8}), 'tps3'));
check('TPS streak reaches three again', has(stepStats(t, [], {tps: 8}), 'tps3'));

t = tracker();
check('10 TPS still counts towards the 8+ streak',
	stepStats(t, [], {tps: 10}).length == 1 &&
	t.state().tps == 1);

// ------------------------------------------------------------- no rotation
t = tracker();
check('四向无转 after a rotation-free full solve',
	has(stepStats(t, [], {norot: true}), 'norot'));
check('四向无转 is not repeated',
	!has(stepStats(t, [], {norot: true}), 'norot'));

t = tracker();
check('TPS tiers are not required for 图鉴大师',
	t.requiredKeys().indexOf('tps10') < 0 &&
	t.requiredKeys().indexOf('tps8') < 0);
check('no-rotation is not a codex technique', t.requiredKeys().indexOf('norot') < 0);

// -------------------------------------------------------------- same solve
t = tracker();
check('全家桶', has(step(t, ['zbls', 'zbll']), 'fullhouse'));
t = tracker();
check('OLS + ZBLL is not 全家桶', !has(step(t, ['ols', 'zbll']), 'fullhouse'));
t = tracker();
check('三连击', has(step(t, ['xc', 'ols', 'zbll']), 'triple'));
t = tracker();
check('two techniques are not 三连击', !has(step(t, ['xc', 'zbll']), 'triple'));

// ----------------------------------------------------------------- codex
t = tracker();
var required = t.requiredKeys();
var codex = [];
required.forEach(function(key) {
	codex = step(t, [key]);
});
check('图鉴大师 after every technique kind', has(codex, 'codex'));
check('图鉴大师 is not repeated', !has(step(t, ['xc']), 'codex'));
check('codex progress is complete', t.progress('codex').cur == t.progress('codex').target);

// -------------------------------------------------- tips as achievements
t = tracker();
t.step(['zbll', 'tps10'], {tps: 10});
check('a technique card is collected', t.progress('zbll').cur == 1 &&
	t.progress('zbll').target == 1);
check('a TPS tip is collected too', t.progress('tps10').cur == 1);
check('a TPS tip does not inflate the combo', t.state().hot == 1);
check('a TPS tip does not inflate the codex', t.progress('codex').cur == 1);
check('catalog exposes technique groups', t.catalog().some(function(def) {
	return def.key == 'zbll' && def.group == 'technique';
}));

// ------------------------------------------------------------ reset all
t = tracker();
step(t, ['zbll']);
step(t, ['zbll']);
step(t, ['zbll']);
check('reset test starts with unlocked progress',
	t.progress('zb3').cur == 3 && t.progress('zbll').cur == 1);
t.reset();
check('reset clears counters and collected cards',
	t.state().solves == 0 && t.progress('zb3').cur == 0 &&
	t.progress('zbll').cur == 0);
check('reset clears the serialized save', (function() {
	var saved = t.serialize();
	return saved.solves == 0 && !Object.keys(saved.seen).length &&
		!Object.keys(saved.unlocked).length;
})());
step(t, ['zbll']);
check('achievements can unlock again after reset',
	t.progress('zbll').cur == 1 && t.state().seen.zbll == 1);

// ------------------------------------------------------- staged milestones
t = copilot.makeAchievementTracker();
var first = step(t, [], {time: 15000});
check('the first solve unlocks its first stage', has(first, 'solves1'));
check('one fast solve unlocks the best-time stages',
	has(first, 'best1') && has(first, 'best2') && has(first, 'best3'));

t = copilot.makeAchievementTracker();
t.load({solves: 100, unlocked: {}});
check('a later stage is hidden before its predecessor',
	t.progress('solves2').cur == 0 && t.progress('solves3').cur == 0);
var staged = step(t, []);
check('stages unlock in order',
	staged.indexOf('solves1') >= 0 &&
	staged.indexOf('solves1') < staged.indexOf('solves2'));

t = tracker();
var yesterday = new Date();
yesterday.setDate(yesterday.getDate() - 1);
var y = yesterday.getFullYear();
var m = yesterday.getMonth() + 1;
var d = yesterday.getDate();
var yKey = y + '-' + (m < 10 ? '0' : '') + m + '-' + (d < 10 ? '0' : '') + d;
t.load({solves: 1, unlocked: {solves1: 1}, streak: 2, bestStreak: 2,
	lastDay: yKey});
check('daily practice streak advances',
	has(step(t, [], {time: 10000}), 'streak1'));

t = tracker();
step(t, [], {time: 10000, method: 'roux', oh: true});
check('method and one-handed counters update',
	t.state().roux == 1 && t.state().onehand == 1 &&
	(t.state().methods & 2) != 0);

t = tracker();
step(t, [], {time: 10000, bld: true, virtual: true});
check('virtual blind solves are counted', t.state().bld == 1);

t = tracker();
step(t, ['ollskip']);
check('a lucky skip starts a streak',
	has(step(t, ['pll']), 'luck1') && t.state().luckStreak == 2);

// ------------------------------------------------- custom achievement API
copilot.registerAchievement({
	key: 'solve10',
	group: 'challenge',
	stat: 'solves',
	target: 10
});
t = tracker();
for (var si = 0; si < 8; si++) {
	step(t, []);
}
check('a custom achievement unlocks through its config',
	has(step(t, []), 'solve10'));
check('a custom achievement has progress', t.progress('solve10').cur == 10);
check('a custom achievement joins the panel order',
	t.list().indexOf('solve10') >= 0);

// ------------------------------------------------------------- persistence
t = tracker();
step(t, ['zbll']);
step(t, ['zbll']);
step(t, ['zbll']);
var saved = JSON.parse(JSON.stringify(t.serialize()));
var t2 = tracker();
t2.load(saved);
check('streak survives reload', t2.state().zbll == 3);
check('unlock survives reload', !!t2.state().unlocked.zb3);
check('ZB 狂人 does not re-fire after reload', !has(step(t2, ['zbll']), 'zb3'));
check('progress uses the saved streak', t2.progress('zb3').cur == 3);

t = tracker();
t.load({seen: {zbll: -2}, unlocked: {nope: 1}, hot: 'x'});
check('bad saved data is ignored', !t.state().unlocked.nope && t.state().zbll == 0);

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
