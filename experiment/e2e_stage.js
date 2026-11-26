"use strict";

// Browser test for the CFOP stage stepper that is shown to non-cubers.
//
//  1. the bar is hidden while idle and appears when a solve starts
//  2. it renders one chip per multi-phase step (cf4op -> 7 chips)
//  3. every facelet push moves the chips to the right step
//  4. the multi-phase countdown does not reset the chips
//  5. a finished solve lingers on SOLVED, an aborted one leaves nothing
//  6. switching the multi-phase mode rebuilds the bar, leaving it hides it

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
	var page = await browser.newPage();
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

	// ------------------------------------------------------- technique tips
	var tech = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
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
		function tipState() {
			var tip = $('#copilotStageTip');
			return {
				text: tip.text(),
				on: tip.hasClass('cp-tip-on'),
				combo: tip.hasClass('cp-tip-combo')
			};
		}
		function push(seq) {
			kernel.pushSignal('virtualState', [faceletOf(seq)]);
		}
		var F = {
			scr: 'R U F L D B R2 U2 F2',                 // prog 7
			xc: "B' B B' U2 F2 U2 B",                     // prog 5: cross + 1 pair
			p2: "U B U2 B2 U2 B U'",                      // prog 4
			p3: "U R U'",                                 // prog 3
			sune: "R U R' U R U2 R'",                     // prog 2, EO done
			epll: "R U' R U R U R U' R' U' R2",           // OLL done, corners placed
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
		var before = tipState();
		push(F.xc);
		await wait(150);
		var xc = tipState();
		await wait(3200);
		var xcCombo = tipState();
		push(F.p2);
		await wait(60);
		push(F.p3);
		await wait(60);
		push(F.sune);
		await wait(60);
		push(F.solved);
		await wait(200);
		var zbll = tipState();

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
		push(F.sune);
		await wait(60);
		push(F.epll);
		await wait(60);
		push(F.solved);
		await wait(200);
		var coll = tipState();

		// the next solve must start with no leftover tip
		kernel.pushSignal('timerStatus', -1);
		await wait(150);
		push(F.scr);
		await wait(50);
		kernel.pushSignal('timerStatus', 1);
		await wait(100);
		var fresh = tipState();

		return {
			before: before,
			xc: xc,
			xcCombo: xcCombo,
			zbll: zbll,
			coll: coll,
			fresh: fresh
		};
	});
	console.log('tech tips: ' + JSON.stringify(tech));

	if (tech.before.on) {
		fail('technique tip is visible before any achievement');
	}
	if (!tech.xc.on || tech.xc.text.indexOf('XC') < 0 || tech.xc.text.indexOf('XXC') >= 0) {
		fail('XC tip: ' + JSON.stringify(tech.xc));
	}
	if (!tech.xcCombo.on || !tech.xcCombo.combo || tech.xcCombo.text.indexOf('XC') < 0) {
		fail('XC combo did not persist: ' + JSON.stringify(tech.xcCombo));
	}
	if (!tech.zbll.on || tech.zbll.text.indexOf('ZBLL') < 0) {
		fail('ZBLL tip: ' + JSON.stringify(tech.zbll));
	}
	if (!tech.coll.on || tech.coll.text.indexOf('COLL') < 0) {
		fail('COLL tip: ' + JSON.stringify(tech.coll));
	}
	if (tech.fresh.on) {
		fail('a new solve kept the previous technique tip: ' + JSON.stringify(tech.fresh));
	}

	fatal.forEach(function(m) {
		fail('page error: ' + m);
	});

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
