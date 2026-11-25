"use strict";

// Browser test for 3x3 relay pacing feedback.
//
//  1. a two-cube r3 relay emits one relaySolve event per solved cube
//  2. the body keeps a green pace background while both cubes beat the target
//  3. a slow cumulative pace turns the background red
//
// The page needs `input = v` and a simple relay scramble, so the test can
// finish each cube with a single inverse move.

var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8142;
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

	var relay = await page.evaluate(async function() {
		function wait(ms) {
			return new Promise(function(resolve) {
				setTimeout(resolve, ms);
			});
		}
		function uiState() {
			var glow = $('#copilotRelayGlow');
			var flash = $('#copilotRelayFlash');
			return {
				glow: glow.attr('class') || '',
				glowBg: glow.length ? getComputedStyle(glow[0]).backgroundImage : '',
				glowAnim: glow.length ? getComputedStyle(glow[0]).animationName : '',
				flash: flash.attr('class') || '',
				flashAnim: flash.length ? getComputedStyle(flash[0]).animationName : ''
			};
		}

		kernel.setProp('input', 'v');
		kernel.setProp('scrType', 'r3');
		kernel.setProp('cpRelayTarget', 10);
		kernel.setProp('vrcSpeed', 0);
		await wait(1200);

		// the target field lives in the scramble-settings popup
		$('#scrambleDiv .title input.icon').first().click();
		await wait(100);
		var input = $('#relayTargetInput');
		var targetUi = {
			shown: input.length == 1 && input.is(':visible'),
			initial: input.val()
		};
		input.val('9').change();
		targetUi.applied = kernel.getProp('cpRelayTarget');
		input.val('10').change();
		targetUi.final = kernel.getProp('cpRelayTarget');

		var events = [];
		kernel.regListener('relayTest', 'relaySolve', function(signal, value) {
			events.push(value.slice());
		});

		kernel.pushSignal('scrambling', 'r3');
		kernel.pushSignal('scramble', ['r3', "1) R\n2) U", 3]);
		await wait(200);
		timer.virtual.onkeydown(32, {});
		await wait(100);
		var started = timer.status();

		// first scramble R -> R'
		timer.virtual.onkeydown(75, {});
		await wait(120);
		var afterFirst = {
			status: timer.status(),
			events: events.slice(),
			ui: uiState()
		};
		await wait(650);
		var settledFirst = uiState();

		// second scramble U -> U'
		timer.virtual.onkeydown(70, {});
		await wait(120);
		var afterSecond = {
			status: timer.status(),
			events: events.slice(),
			ui: uiState()
		};

		// force a slow cumulative pace without running another relay
		kernel.pushSignal('timerStatus', -2);
		kernel.pushSignal('relaySolve', [2, 2, 15000, 12000, true]);
		var splitFlash = uiState();
		await wait(650);
		var splitBase = uiState();

		kernel.pushSignal('timerStatus', -2);
		kernel.pushSignal('relaySolve', [2, 2, 25000, 5000, true]);
		var slowFlash = uiState();
		await wait(650);

		return {
			started: started,
			targetUi: targetUi,
			afterFirst: afterFirst,
			settledFirst: settledFirst,
			afterSecond: afterSecond,
			splitFlash: splitFlash,
			splitBase: splitBase,
			slowFlash: slowFlash,
			slowBase: uiState()
		};
	});

	console.log(JSON.stringify(relay, null, 2));
	if (!relay.targetUi.shown) {
		fail('relay target input is not visible for r3');
	}
	if (relay.targetUi.initial != '10') {
		fail('relay target input did not show the current value: ' + relay.targetUi.initial);
	}
	if (relay.targetUi.applied != 9) {
		fail('relay target input did not update cpRelayTarget: ' + relay.targetUi.applied);
	}
	if (relay.targetUi.final != 10) {
		fail('relay target input did not restore cpRelayTarget');
	}
	if (relay.started != -2) {
		fail('relay did not start');
	}
	if (relay.afterFirst.status != 1) {
		fail('timer should still run after the first relay cube');
	}
	if (relay.afterFirst.events.length != 1) {
		fail('expected one relaySolve event after the first cube');
	}
	if (relay.afterSecond.events.length != 2) {
		fail('expected two relaySolve events after the second cube');
	}
	if (relay.afterSecond.status != -1) {
		fail('timer should stop after the final relay cube');
	}
	if (!/relay-glow-on/.test(relay.afterFirst.ui.glow) ||
		!/relay-glow-ok/.test(relay.afterFirst.ui.glow)) {
		fail('fast relay should breathe on the green edge: ' + relay.afterFirst.ui.glow);
	}
	if (relay.afterFirst.ui.glowAnim != 'copilotRelayBreath') {
		fail('relay edge glow is not breathing: ' + relay.afterFirst.ui.glowAnim);
	}
	if (!/radial-gradient/.test(relay.afterFirst.ui.glowBg)) {
		fail('relay glow is not an edge gradient: ' + relay.afterFirst.ui.glowBg);
	}
	if (!/relay-flash-run/.test(relay.afterFirst.ui.flash) ||
		relay.afterFirst.ui.flashAnim != 'copilotRelayFlash') {
		fail('cube completion did not flash: ' + JSON.stringify(relay.afterFirst.ui));
	}
	if (!/relay-glow-ok/.test(relay.settledFirst.glow)) {
		fail('fast relay should settle on the green edge: ' + relay.settledFirst.glow);
	}
	if (!/relay-glow-ok/.test(relay.splitBase.glow)) {
		fail('fast cumulative pace should stay green when only the split is slow: ' + relay.splitBase.glow);
	}
	if (!/relay-flash-bad/.test(relay.splitFlash.flash)) {
		fail('slow split should flash red: ' + relay.splitFlash.flash);
	}
	if (!/relay-flash-ok/.test(relay.slowFlash.flash)) {
		fail('fast split should flash green even when the cumulative pace is slow: ' + relay.slowFlash.flash);
	}
	if (!/relay-glow-bad/.test(relay.slowBase.glow)) {
		fail('slow relay should settle on the red edge: ' + relay.slowBase.glow);
	}

	fatal.forEach(function(m) {
		fail('page error: ' + m);
	});

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
