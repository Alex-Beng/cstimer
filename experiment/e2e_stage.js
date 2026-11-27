"use strict";

// Browser test for the CFOP stage stepper that is shown to non-cubers.
//
//  1. the bar is hidden while idle and appears when a solve starts
//  2. it renders one chip per multi-phase step (cf4op -> 7 chips)
//  3. every facelet push moves the chips to the right step
//  4. the multi-phase countdown does not reset the chips
//  5. a finished solve lingers on SOLVED, an aborted one leaves nothing
//  6. switching the multi-phase mode rebuilds the bar, leaving it hides it
//  7. relay resets the stepper and achievements for every new cube
//  8. achievements pop up as a toast, then fade while the combo line stays
//  9. a completed solve unlocks a cross-solve achievement and updates the panel

var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8147;
var failures = 0;

function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
}

var MIME = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json',
	'.webmanifest': 'application/manifest+json'
};

var server = http.createServer(function(req, res) {
	var url = decodeURIComponent(req.url.split('?')[0]);
	if (url == '/') {
		url = '/index.html';
	}
	var file = path.join(root, url.replace(/^\/+/, ''));
	if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
		res.writeHead(404);
		res.end('not found');
		return;
	}
	res.writeHead(200, {'Content-Type': MIME[path.extname(file)] || 'application/octet-stream'});
	fs.createReadStream(file).pipe(res);
});

(async function() {
	await new Promise(function(resolve) {
		server.listen(port, '0.0.0.0', function() {
			resolve();
		});
	});

	var browser = await pw.chromium.launch();
	var page = await browser.newPage({locale: 'en-US'});
	var fatal = [];
	page.on('pageerror', function(e) {
		var msg = e.message || '';
		if (/ServiceWorker|sw\.js|Wake Lock|bad HTTP response code/i.test(msg)) {
			return;
		}
		fatal.push(msg);
	});

	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var out = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
		}
		var SHAPES = {
			cf4op: ['Cross', 'F1', 'F2', 'F3', 'F4', 'OLL', 'PLL'],
			cfop: ['Cross', 'F2L', 'OLL', 'PLL'],
			roux: ['FB', 'SB', 'CMLL', 'L6E']
		};

		function barState() {
			var bar = $('#copilotStageBar');
			var chips = bar.find('.cp-stage-chip');
			var state = {
				exists: bar.length > 0,
				on: bar.hasClass('cp-stage-on'),
				label: bar.find('.cp-stage-label').text(),
				texts: [],
				done: [],
				cur: -1
			};
			chips.each(function(i) {
				state.texts.push($(this).text());
				state.done.push($(this).hasClass('cp-done'));
				if ($(this).hasClass('cp-cur')) {
					state.cur = i;
				}
			});
			return state;
		}

		function faceletOf(moves) {
			var cc = new mathlib.CubieCube();
			moves.split(' ').forEach(function(m) {
				cc.selfMoveStr(m);
			});
			cc.ori = 0;
			return cc.toFaceCube();
		}

		function pushFacelet(facelet) {
			kernel.pushSignal('virtualState', [facelet]);
		}

		// the state implied by a facelet, computed independently of the UI
		function expected(texts, method, facelet) {
			var base = texts.length;
			var progress = cubeutil.getProgress(facelet, method);
			return Math.max(0, Math.min(base, base - progress));
		}

		var STATES = {
			mixed: faceletOf('R U F L D B R2 U2 F2'),
			oll: faceletOf('R U R\' U R U2 R\''),          // Sune: F2L done
			pll: faceletOf('R U R\' U\' R\' F R2 U\' R\' U\' R U R\' F\''), // T perm
			solved: mathlib.SOLVED_FACELET
		};

		kernel.setProp('input', 'v');
		kernel.setProp('scrType', '333');
		kernel.setProp('vrcMP', 'cf4op');
		await wait(400);
		var idle = barState();

		// a solve starts on the virtual/relay cube
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		var armed = barState();

		var order = ['mixed', 'oll', 'pll', 'solved'];
		var seen = [];
		var regressed = null;
		for (var i = 0; i < order.length; i++) {
			pushFacelet(STATES[order[i]]);
			await wait(40);
			seen.push({
				name: order[i],
				state: barState(),
				expect: expected(SHAPES.cf4op, 'cf4op', STATES[order[i]])
			});
			if (order[i] == 'oll') {
				// a mid-alg facelet can show a temporarily broken cross;
				// the stepper must keep the furthest stage reached
				pushFacelet(STATES.mixed);
				await wait(40);
				regressed = barState();
			}
		}

		// the phase countdown is not a new solve: the chips must not reset
		kernel.pushSignal('timerStatus', 6);
		await wait(40);
		var afterCountdown = barState();

		// switching the multi-phase mode rebuilds from the last facelet
		kernel.setProp('vrcMP', 'cfop');
		await wait(100);
		var switched = barState();
		var switchedExpect = expected(SHAPES.cfop, 'cfop', STATES.solved);

		kernel.setProp('vrcMP', 'cf4op');
		await wait(100);
		var backToCf4op = barState();

		// finishing keeps SOLVED on screen for a moment
		pushFacelet(STATES.solved);
		await wait(40);
		kernel.pushSignal('timerStatus', -1);
		await wait(120);
		var finished = barState();
		await wait(2900);
		var finishedCleared = barState();

		// an aborted attempt must not leave anything behind
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		pushFacelet(STATES.mixed);
		await wait(40);
		var abortedBefore = barState();
		kernel.pushSignal('timerStatus', -1);
		await wait(120);
		var abortedAfter = barState();

		// a single-phase mode has nothing to show
		kernel.setProp('vrcMP', 'n');
		await wait(100);
		var singlePhase = barState();

		// a relay scramble uses the same stepper
		kernel.setProp('scrType', 'r3');
		kernel.setProp('vrcMP', 'cf4op');
		await wait(400);
		kernel.pushSignal('timerStatus', 1);
		await wait(80);
		var relay = barState();

		// one cube done in a relay: the next cube starts from Cross again
		pushFacelet(STATES.solved);
		await wait(60);
		var relaySolved = barState();
		kernel.pushSignal('relaySolve', [1, 2, 5000, 5000, false]);
		await wait(80);
		var relayReset = barState();
		var relayCombo = $('#copilotStageTip').text();

		return {
			idle: idle,
			armed: armed,
			seen: seen,
			regressed: regressed,
			afterCountdown: afterCountdown,
			switched: switched,
			switchedExpect: switchedExpect,
			backToCf4op: backToCf4op,
			finished: finished,
			finishedCleared: finishedCleared,
			abortedBefore: abortedBefore,
			abortedAfter: abortedAfter,
			singlePhase: singlePhase,
			relay: relay,
			relaySolved: relaySolved,
			relayReset: relayReset,
			relayCombo: relayCombo,
			shapes: SHAPES
		};
	});

	console.log(JSON.stringify(out, null, 2));

	function checkIdx(name, state, idx) {
		var n = state.texts.length;
		if (n == 0) {
			fail(name + ': no chips rendered');
			return;
		}
		if (state.cur != idx && !(idx >= n && state.cur == -1)) {
			fail(name + ': current chip ' + state.cur + ', expected ' + idx);
		}
		for (var i = 0; i < n; i++) {
			var wantDone = i < idx;
			if (!!state.done[i] != wantDone) {
				fail(name + ': chip ' + i + ' done=' + state.done[i] + ', expected ' + wantDone);
			}
		}
	}

	if (!out.idle.exists) {
		fail('stage bar was never created');
	}
	if (out.idle.on) {
		fail('stage bar is visible while idle');
	}
	if (out.idle.texts.join(',') != out.shapes.cf4op.join(',')) {
		fail('idle chips: ' + out.idle.texts.join(','));
	}
	if (!out.armed.on) {
		fail('stage bar did not appear when the solve started');
	}
	checkIdx('start', out.armed, 0);
	if (out.armed.label != 'CFOP') {
		fail('label at start: ' + out.armed.label);
	}

	out.seen.forEach(function(step) {
		if (step.state.texts.join(',') != out.shapes.cf4op.join(',')) {
			fail(step.name + ': wrong chips ' + step.state.texts.join(','));
		}
		checkIdx(step.name, step.state, step.expect);
		if (step.expect >= out.shapes.cf4op.length) {
			if (step.state.label != 'SOLVED') {
				fail(step.name + ': label is ' + step.state.label);
			}
		}
	});
	if (!(out.seen[0].expect == 0)) {
		fail('a scrambled cube should start on Cross, got ' + out.seen[0].expect);
	}
	checkIdx('mid-alg rollback', out.regressed, out.seen[1].expect);

	checkIdx('phase countdown', out.afterCountdown, out.seen[out.seen.length - 1].expect);

	if (out.switched.texts.join(',') != out.shapes.cfop.join(',')) {
		fail('switch to cfop chips: ' + out.switched.texts.join(','));
	}
	checkIdx('switch to cfop', out.switched, out.switchedExpect);

	if (out.backToCf4op.texts.join(',') != out.shapes.cf4op.join(',')) {
		fail('switch back chips: ' + out.backToCf4op.texts.join(','));
	}

	if (!out.finished.on || out.finished.label != 'SOLVED') {
		fail('finished solve did not linger on SOLVED: ' + JSON.stringify(out.finished));
	}
	if (out.finishedCleared.on) {
		fail('finished solve never cleared');
	}

	if (!out.abortedBefore.on) {
		fail('stage bar missing during the aborted solve');
	}
	if (out.abortedAfter.on) {
		fail('aborted solve left the stage bar behind');
	}

	if (out.singlePhase.on || out.singlePhase.texts.length > 0) {
		fail('single-phase mode still shows a stepper: ' + JSON.stringify(out.singlePhase));
	}

	if (!out.relay.on || out.relay.texts.length != out.shapes.cf4op.length) {
		fail('relay scramble does not show the stepper: ' + JSON.stringify(out.relay));
	}
	if (!out.relaySolved.on || out.relaySolved.label != 'SOLVED') {
		fail('relay cube did not reach SOLVED: ' + JSON.stringify(out.relaySolved));
	}
	if (!out.relayReset.on || out.relayReset.label != 'CFOP') {
		fail('relay did not reset the stepper: ' + JSON.stringify(out.relayReset));
	}
	checkIdx('relay next cube', out.relayReset, 0);
	if (out.relayCombo) {
		fail('relay kept the previous cube technique combo: ' + out.relayCombo);
	}

	// ------------------------------------------------------- technique tips
	var tech = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
		}
		// queued technique tips may need a moment before the wanted one shows
		async function waitToast(part, ms) {
			var end = Date.now() + ms;
			while (Date.now() < end) {
				if ($('#copilotTechToast .cp-toast-label').text().indexOf(part) >= 0) {
					return true;
				}
				await wait(50);
			}
			return false;
		}
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
		function invAlg(alg) {
			return alg.trim().split(/\s+/).reverse().map(function(m) {
				if (m.slice(-1) == "'") {
					return m.slice(0, -1);
				}
				return m.slice(-1) == '2' ? m : m + "'";
			}).join(' ');
		}
		function comboState() {
			var tip = $('#copilotStageTip');
			return {
				text: tip.text(),
				on: tip.hasClass('cp-tip-on'),
				combo: tip.hasClass('cp-tip-combo')
			};
		}
		function toastState() {
			var toast = $('#copilotTechToast');
			return {
				exists: toast.length > 0,
				cls: toast.attr('class') || '',
				label: toast.find('.cp-toast-label').text(),
				desc: toast.find('.cp-toast-desc').text(),
				open: toast.hasClass('cp-toast-in') && !toast.hasClass('cp-toast-out'),
				faded: !toast.hasClass('cp-toast-in') && !toast.hasClass('cp-toast-out')
			};
		}
		function state() {
			return {
				toast: toastState(),
				combo: comboState()
			};
		}
		function achState() {
			var toast = $('#copilotAchToast');
			return {
				toast: {
					exists: toast.length > 0,
					cls: toast.attr('class') || '',
					label: toast.find('.cp-ach-label').text(),
					desc: toast.find('.cp-ach-desc').text(),
					open: toast.hasClass('cp-ach-in') && !toast.hasClass('cp-ach-out')
				},
				panel: $('#copilotAchPanel').text(),
				comboBadge: $('#copilotStageBar .cp-stage-combo').text()
			};
		}
		function push(seq) {
			kernel.pushSignal('virtualState', [faceletOf(seq)]);
		}
		// a real COLL case (AS 2): EO done after F2L, corners twisted and permuted
		var collAlg = "R2 D R' U R D' R' U R' U' R U' R'";
		var epllAlg = "R U' R U R U R U' R' U' R2";
		var F = {
			scr: 'R U F L D B R2 U2 F2',                 // prog 7
			xc: "B' B B' U2 F2 U2 B",                     // prog 5: cross + 1 pair
			p2: "U B U2 B2 U2 B U'",                      // prog 4
			p3: "U R U'",                                 // prog 3
			sune: "R U R' U R U2 R'",                     // prog 2, EO done
			collCase: epllAlg + ' ' + invAlg(collAlg),    // prog 2, EO done
			collDone: epllAlg,                            // corners placed, EPLL left
			solved: mathlib.SOLVED_FACELET
		};

		kernel.setProp('scrType', '333');
		kernel.setProp('vrcMP', 'cf4op');
		kernel.pushSignal('timerStatus', -1);
		await wait(200);

		// XC, then a ZBLL finish
		push(F.scr);
		await wait(50);
		kernel.pushSignal('timerStatus', 1);
		await wait(80);
		var before = state();
		push(F.xc);
		await wait(150);
		var xc = state();
		await wait(3200);
		var xcCombo = state();
		push(F.p2);
		await wait(60);
		push(F.p3);
		await wait(60);
		push(F.sune);
		await wait(60);
		push(F.solved);
		await waitToast('ZBLL', 12000);
		var zbll = state();

		// a fresh solve: COLL + EPLL
		kernel.pushSignal('timerStatus', -1);
		await wait(150);
		push(F.scr);
		await wait(50);
		kernel.pushSignal('timerStatus', 1);
		await wait(80);
		push(F.xc);
		await wait(60);
		push(F.p2);
		await wait(60);
		push(F.p3);
		await wait(60);
		push(F.collCase);
		await wait(60);
		push(F.collDone);
		await waitToast('COLL', 12000);
		var collAtOll = state();
		push(F.solved);
		await waitToast('EPLL', 12000);
		var coll = state();

		// a joined pair in the U layer, inserted by three moves.  The
		// virtual move stream is what tells the tip where "three moves ago"
		// is, so a rotation cannot shorten that window.
		// This synthetic replay ends on a solved state; mark it rotated so it
		// cannot also unlock the rotation-free achievement.
		kernel.pushSignal('virtualMove', ['y']);
		push("R U R'");
		await wait(40);
		kernel.pushSignal('virtualMove', ['R']);
		push("R U R'");
		await wait(60);
		kernel.pushSignal('virtualMove', ['R']);
		push("R U R' R");
		await wait(60);
		kernel.pushSignal('virtualMove', ["U'"]);
		push("R U R' R U'");
		await wait(60);
		kernel.pushSignal('virtualMove', ["R'"]);
		push("R U R' R U' R'");
		await wait(200);
		var basic = state();
		push(F.solved);
		await wait(100);

		// the next solve must start with no leftover tip
		kernel.pushSignal('timerStatus', -1);
		await wait(150);
		kernel.pushSignal('time', [10000]);
		await wait(400);
		var ach = achState();
		push(F.scr);
		await wait(50);
		kernel.pushSignal('timerStatus', 1);
		await wait(100);
		var fresh = state();
		var freshAch = achState();

		// A six move F2L that only reaches the standard shape for its last
		// three moves.  The pair is joined after move 3 but not after move 2,
		// so this is a conversion and must not report Basic F2L.
		var convStart = 'RRDLUBDFLFRBURUFRRBLDFFLLFDBDRDDDLDBFBLFLRFLURUUBBUUBU';
		var convMoves = ['R', 'U', "L'", 'U2', "R'", 'L'];
		kernel.pushSignal('timerStatus', -1);
		await wait(120);
		push(convStart);
		await wait(50);
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		for (var c = 0; c < convMoves.length; c++) {
			kernel.pushSignal('virtualMove', [convMoves[c]]);
			push(convStart + ' ' + convMoves.slice(0, c + 1).join(' '));
			await wait(60);
		}
		await wait(250);
		var converted = state();

		return {
			before: before,
			xc: xc,
			xcCombo: xcCombo,
			zbll: zbll,
			collAtOll: collAtOll,
			coll: coll,
			basic: basic,
			converted: converted,
			ach: ach,
			fresh: fresh,
			freshAch: freshAch
		};
	});
	console.log('tech tips: ' + JSON.stringify(tech));

	if (!tech.before.toast.exists || tech.before.toast.cls || tech.before.combo.on) {
		fail('technique tip is visible before any achievement: ' + JSON.stringify(tech.before));
	}
	if (!tech.xc.toast.open || tech.xc.toast.label.indexOf('XC') < 0 ||
			tech.xc.toast.label.indexOf('XXC') >= 0 || tech.xc.toast.desc.indexOf('cross') < 0) {
		fail('XC achievement toast: ' + JSON.stringify(tech.xc));
	}
	if (!tech.xc.combo.combo || tech.xc.combo.text.indexOf('XC') < 0) {
		fail('XC combo did not update: ' + JSON.stringify(tech.xc.combo));
	}
	if (!tech.xcCombo.toast.faded || !tech.xcCombo.combo.on) {
		fail('XC toast did not fade while the combo stayed: ' + JSON.stringify(tech.xcCombo));
	}
	if (!tech.zbll.toast.open || tech.zbll.toast.label.indexOf('ZBLL') < 0) {
		fail('ZBLL achievement toast: ' + JSON.stringify(tech.zbll));
	}
	if (!tech.coll.toast.open || tech.coll.toast.label.indexOf('EPLL') < 0 ||
			tech.coll.combo.text.indexOf('COLL') < 0 ||
			tech.coll.combo.text.indexOf('EPLL') < 0) {
		fail('EPLL achievement toast after COLL: ' + JSON.stringify(tech.coll));
	}
	if (!tech.collAtOll.toast.open || tech.collAtOll.toast.label.indexOf('COLL') < 0 ||
			tech.collAtOll.combo.text.indexOf('COLL') < 0) {
		fail('COLL did not pop when OLL completed: ' + JSON.stringify(tech.collAtOll));
	}
	if (tech.basic.combo.text.indexOf('Basic F2L') < 0) {
		fail('basic F2L tip missing: ' + JSON.stringify(tech.basic));
	}
	if (tech.converted.combo.text.indexOf('Basic F2L') >= 0) {
		fail('a converted F2L was reported as Basic F2L: ' +
			JSON.stringify(tech.converted));
	}
	if (tech.fresh.toast.cls || tech.fresh.combo.on) {
		fail('a new solve kept the previous technique tip: ' + JSON.stringify(tech.fresh));
	}
	if (!tech.ach.toast.open || tech.ach.toast.label.indexOf('Triple Combo') < 0) {
		fail('achievement toast did not pop: ' + JSON.stringify(tech.ach));
	}
	if (tech.ach.panel.indexOf('Triple Combo') < 0 ||
			tech.ach.panel.indexOf('✓') < 0) {
		fail('achievement panel was not updated: ' + JSON.stringify(tech.ach));
	}
	if (tech.freshAch.toast.open || tech.freshAch.toast.cls) {
		fail('a new solve kept the previous achievement toast: ' + JSON.stringify(tech.freshAch));
	}

	// ------------------------------------------------- TPS / rotation-free
	var speed = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
		}
		async function waitToast(part, ms) {
			var end = Date.now() + ms;
			while (Date.now() < end) {
				if ($('#copilotTechToast .cp-toast-label').text().indexOf(part) >= 0) {
					return true;
				}
				await wait(50);
			}
			return false;
		}
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
		function panel() {
			return $('#copilotAchPanel').text();
		}
		function combo() {
			return $('#copilotStageTip').text();
		}
		var scr = 'R U F L D B R2 U2 F2';
		kernel.setProp('scrType', '333');
		kernel.setProp('vrcMP', 'cf4op');
		kernel.pushSignal('timerStatus', -1);
		await wait(150);

		// a whole-cube rotation disqualifies the solve from the no-rotation tip
		kernel.pushSignal('virtualState', [faceletOf(scr)]);
		await wait(40);
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		kernel.pushSignal('virtualMove', ['x']);
		await wait(30);
		kernel.pushSignal('virtualState', [mathlib.SOLVED_FACELET]);
		await wait(80);
		kernel.pushSignal('timerStatus', -1);
		await wait(80);
		kernel.pushSignal('time', [10000]);
		await wait(300);
		var rotatedCombo = combo();
		var rotatedPanel = panel();

		// 10 quick face turns trigger a TPS tip; no rotation keeps it clean
		kernel.pushSignal('timerStatus', -1);
		await wait(150);
		kernel.pushSignal('virtualState', [faceletOf(scr)]);
		await wait(40);
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		for (var i = 0; i < 10; i++) {
			kernel.pushSignal('virtualMove', ['R']);
			await wait(40);
		}
		var sawTps = await waitToast('TPS', 3000);
		kernel.pushSignal('virtualState', [mathlib.SOLVED_FACELET]);
		await wait(100);
		kernel.pushSignal('timerStatus', -1);
		await wait(80);
		kernel.pushSignal('time', [10000]);
		await wait(500);
		var fastCombo = combo();
		var fastPanel = panel();

		// a single-phase timer has no stepper, but still scores TPS
		kernel.setProp('vrcMP', 'n');
		await wait(120);
		kernel.pushSignal('timerStatus', -1);
		await wait(150);
		kernel.pushSignal('virtualState', [faceletOf(scr)]);
		await wait(40);
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		for (var j = 0; j < 10; j++) {
			kernel.pushSignal('virtualMove', ['R']);
			await wait(40);
		}
		kernel.pushSignal('virtualState', [mathlib.SOLVED_FACELET]);
		await wait(100);
		kernel.pushSignal('timerStatus', -1);
		await wait(80);
		kernel.pushSignal('time', [10000]);
		await wait(500);

		// a bluetooth cube has no virtualMove stream; the centre frame is
		// used instead, so a rotation-free smart solve still counts
		kernel.setProp('scrType', '333');
		kernel.pushSignal('timerStatus', -1);
		await wait(120);
		kernel.pushSignal('giikerState', [faceletOf(scr)]);
		await wait(60);
		kernel.pushSignal('timerStatus', 1);
		await wait(60);
		kernel.pushSignal('giikerState', [mathlib.SOLVED_FACELET]);
		await wait(300);
		var smartCombo = $('#copilotTechToast').text();

		return {
			rotatedCombo: rotatedCombo,
			rotatedPanel: rotatedPanel,
			sawTps: sawTps,
			combo: fastCombo,
			panel: fastPanel,
			singlePanel: panel(),
			smartCombo: smartCombo
		};
	});
	console.log('speed/norot: ' + JSON.stringify(speed));

	if (speed.rotatedCombo.indexOf('No Rot') >= 0) {
		fail('a rotated solve showed the no-rotation tip: ' + speed.rotatedCombo);
	}
	if (speed.rotatedPanel.indexOf('Four-Way F2L0/1') < 0) {
		fail('a rotated solve unlocked the no-rotation achievement: ' + speed.rotatedPanel);
	}
	if (!speed.sawTps) {
		fail('fast virtual turns did not show a TPS tip');
	}
	if (speed.combo.indexOf('No Rot') < 0) {
		fail('a rotation-free solve did not show the no-rotation tip: ' + speed.combo);
	}
	if (speed.panel.indexOf('Four-Way F2L✓') < 0) {
		fail('the rotation-free achievement did not unlock: ' + speed.panel);
	}
	if (speed.panel.indexOf('Speed Demon1/3') < 0) {
		fail('the fast solve did not count towards the TPS streak: ' + speed.panel);
	}
	if (speed.singlePanel.indexOf('Speed Demon2/3') < 0) {
		fail('a single-phase solve did not count towards the TPS streak: ' + speed.singlePanel);
	}
	if (speed.smartCombo.indexOf('No Rot') < 0) {
		fail('a smart-cube solve could not complete No Rot: ' + speed.smartCombo);
	}

	// ---------------------------------------------------- achievement browser
	var book = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
		}
		async function waitToast(part, ms) {
			var end = Date.now() + ms;
			while (Date.now() < end) {
				if ($('#copilotTechToast .cp-toast-label').text().indexOf(part) >= 0) {
					return true;
				}
				await wait(50);
			}
			return false;
		}
		function snapshot() {
			var box = $('#copilotAchBook');
			return {
				open: box.hasClass('cp-book-open'),
				count: box.find('.cp-book-card').length,
				resetBtn: box.find('.cp-book-reset').length,
				locked: box.find('.cp-book-card.cp-book-locked').length,
				names: box.find('.cp-book-card-name').map(function() {
					return $(this).text();
				}).get(),
				groups: box.find('.cp-book-group-title').map(function() {
					return $(this).text();
				}).get(),
				countText: box.find('.cp-book-count').text(),
				filter: box.find('.cp-book-tab-on').text()
			};
		}
		var summary = $('#copilotAchPanel .cp-ach-summary').text();
		var browse = $('#copilotAchPanel .cp-ach-browse').length;
		var toolOptions = $('#toolsDiv select option').map(function() {
			return $(this).text();
		}).get();
		var api = typeof copilotUI.openAchievements;
		$('.cp-ach-browse').click();
		await wait(180);
		var all = snapshot();
		var grid = $('#copilotAchBook .cp-book-grid');
		var gridEl = grid[0];
		gridEl.scrollTop = gridEl.scrollHeight;
		await wait(80);
		var gridRect = gridEl.getBoundingClientRect();
		var winRect = $('#copilotAchBook .cp-book-window')[0].getBoundingClientRect();
		var lastCard = grid.find('.cp-book-card').last()[0].getBoundingClientRect();
		var bookLayout = {
			scrollable: gridEl.scrollHeight > gridEl.clientHeight,
			insideWindow: gridRect.bottom <= winRect.bottom + 1,
			lastCardVisible: lastCard.bottom <= gridRect.bottom + 1
		};
		$('#copilotAchBook .cp-book-tab[data-filter="technique"]').click();
		await wait(100);
		var tech = snapshot();
		$('#copilotAchBook .cp-book-card[data-key="zbll"]').click();
		var sawZbll = await waitToast('ZBLL', 8000);
		var techPreview = $('#copilotTechToast .cp-toast-label').text();
		$('#copilotAchBook .cp-book-tab[data-filter="speed"]').click();
		await wait(100);
		$('#copilotAchBook .cp-book-card[data-key="speed"]').click();
		await wait(120);
		var metaPreview = {
			label: $('#copilotAchToast .cp-ach-label').text(),
			kicker: $('#copilotAchToast .cp-ach-kicker').text(),
			open: $('#copilotAchToast').hasClass('cp-ach-in')
		};
		var resetOk = copilotUI.resetAchievements();
		await wait(120);
		var afterReset = {
			ok: resetOk,
			count: $('#copilotAchBook .cp-book-count').text(),
			summary: $('#copilotAchPanel .cp-ach-summary').text()
		};
		$('#copilotAchBook .cp-book-close').click();
		await wait(100);
		return {
			summary: summary,
			browse: browse,
			toolOptions: toolOptions,
			api: api,
			all: all,
			layout: bookLayout,
			tech: tech,
			sawZbll: sawZbll,
			techPreview: techPreview,
			metaPreview: metaPreview,
			afterReset: afterReset,
			closed: !$('#copilotAchBook').hasClass('cp-book-open')
		};
	});
	console.log('achievement book: ' + JSON.stringify(book));

	if (!book.all.open) {
		fail('achievement browser did not open');
	}
	if (!/\d+\/\d+/.test(book.summary)) {
		fail('achievement summary is missing: ' + book.summary);
	}
	if (book.browse != 1) {
		fail('achievement browse button is missing');
	}
	if (book.toolOptions.join('|').indexOf('Copilot achievements') < 0) {
		fail('Tools menu has no achievements entry: ' + book.toolOptions.join('|'));
	}
	if (book.api != 'function') {
		fail('openAchievements API is missing');
	}
	if (book.all.count < 30) {
		fail('achievement browser is missing cards: ' + book.all.count);
	}
	if (book.all.resetBtn != 1) {
		fail('achievement browser has no reset-all button');
	}
	if (book.all.locked < 1 ||
			book.all.names.join('|').indexOf('???') < 0) {
		fail('locked stage cards are not hidden: ' + JSON.stringify(book.all));
	}
	if (book.all.names.join('|').indexOf('II · Limit Breaker') < 0) {
		fail('unlocked stage I did not reveal stage II');
	}
	if (book.all.groups.join('|').indexOf('Advanced techniques') < 0 ||
			book.all.groups.join('|').indexOf('Combos') < 0 ||
			book.all.groups.join('|').indexOf('Speed') < 0 ||
			book.all.groups.join('|').indexOf('Challenges') < 0 ||
			book.all.groups.join('|').indexOf('Milestones') < 0 ||
			book.all.groups.join('|').indexOf('Luck') < 0 ||
			book.all.groups.join('|').indexOf('Methods') < 0) {
		fail('achievement browser groups: ' + book.all.groups.join('|'));
	}
	['ZBLL', 'TPS 10', 'Lightspeed', 'Four-Way F2L'].forEach(function(name) {
		if (book.all.names.indexOf(name) < 0) {
			fail('achievement browser is missing ' + name);
		}
	});
	if (!/^\d+\/\d+$/.test(book.all.countText)) {
		fail('achievement browser count: ' + book.all.countText);
	}
	if (!book.layout.scrollable || !book.layout.insideWindow ||
			!book.layout.lastCardVisible) {
		fail('achievement browser grid is clipped: ' + JSON.stringify(book.layout));
	}
	if (book.tech.filter != 'Advanced techniques' || book.tech.count != 21) {
		fail('technique filter: ' + JSON.stringify(book.tech));
	}
	if (!book.sawZbll || book.techPreview.indexOf('ZBLL') < 0) {
		fail('technique preview: ' + JSON.stringify(book));
	}
	if (!book.metaPreview.open || book.metaPreview.label != 'Lightspeed' ||
			book.metaPreview.kicker.toLowerCase().indexOf('preview') < 0) {
		fail('achievement preview: ' + JSON.stringify(book.metaPreview));
	}
	var total = book.all.countText.split('/')[1];
	if (!book.afterReset.ok || book.afterReset.count != '0/' + total ||
			book.afterReset.summary.indexOf('0/' + total) < 0) {
		fail('achievement reset: ' + JSON.stringify(book.afterReset));
	}
	if (!book.closed) {
		fail('achievement browser did not close');
	}

	fatal.forEach(function(m) {
		fail('page error: ' + m);
	});

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
