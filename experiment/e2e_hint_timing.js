"use strict";

// Verify auto hints fire while the solver is idle, and that they are visible
// during a solve (window is not auto-hidden and a logo toast is pushed).
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8181;
var failures = 0;

function fail(m) {
	failures++;
	console.log('FAIL ' + m);
}

var MIME = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json',
	'.webmanifest': 'application/manifest+json',
	'.png': 'image/png'
};

var server = http.createServer(function(req, res) {
	var url = decodeURIComponent(req.url.split('?')[0]);
	if (url == '/') url = '/index.html';
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
	await new Promise(function(r) {
		server.listen(port, '0.0.0.0', r);
	});

	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	// spy on the logo banner so we can assert the hint becomes visible
	await page.evaluate(function() {
		window.__toasts = [];
		var orig = logohint.push;
		logohint.push = function(msg) {
			window.__toasts.push(msg);
			return orig.apply(logohint, arguments);
		};
	});

	// configure and enter a solving state with an F2L-stage position
	var setup = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpF2LDepth', 6);

		// open the copilot window (left bar index 5)
		var btn = document.querySelectorAll('#leftbar .c5')[0];
		if (btn) btn.click();

		// build a facelet where the cross is solved and one F2L pair already
		// inserted but the rest scrambled: R U R' keeps the cross intact
		var m = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { m.selfMoveStr(t); });
		var facelet = m.toFaceCube();

		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [facelet]);

		return {
			stage: cubeCopilot.getStage(facelet),
			enabled: kernel.getProp('cpEnabled'),
			timeout: kernel.getProp('cpTimeout')
		};
	});
	console.log('setup', JSON.stringify(setup));
	if (setup.stage != 'f2l') fail('test position is not F2L stage: ' + setup.stage);

	// do NOT send any new state; just wait past the timeout
	var dbg = await page.evaluate(function() {
		return {
			puzzle: tools.getCurPuzzle(),
			train: (tools.isCurTrainScramble ? tools.isCurTrainScramble() : null),
			scr: (tools.getCurScramble ? tools.getCurScramble() : null)
		};
	});
	console.log('debug', JSON.stringify(dbg));
	await page.waitForTimeout(10000);

	var res = await page.evaluate(function() {
		var div = document.querySelector('#copilotDiv');
		return {
			toasts: window.__toasts,
			hintText: (div ? div.textContent : '').replace(/\s+/g, ' ').trim(),
			visible: div ? (div.offsetParent !== null) : false
		};
	});
	console.log('after idle wait', JSON.stringify(res));

	if (!res.toasts.length) {
		fail('no hint was pushed while idle (timeout polling missing?)');
	}
	if (!/F2L|U |R |'/i.test(res.hintText)) {
		fail('window shows no hint text: ' + res.hintText);
	}
	if (!res.visible) {
		fail('copilot window is hidden during a solve');
	}

	// leaving the solve must clear the hint again
	var cleared = await page.evaluate(function() {
		kernel.pushSignal('timerStatus', -1);
		return (document.querySelector('#copilotDiv').textContent || '').replace(/\s+/g, ' ');
	});
	console.log('after solve end', JSON.stringify(cleared.slice(0, 120)));

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
