"use strict";

// Verify auto hints fire while the solver is idle AND remain visible during a
// solve, when csTimer fades #leftbar out and auto-hides module windows.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8191;
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

	var setup = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpF2LDepth', 6);

		var m = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { m.selfMoveStr(t); });
		var facelet = m.toFaceCube();

		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [facelet]);

		// emulate what csTimer does when a solve starts: hide module windows
		// and fade the left bar (which carries the logo banner)
		kernel.ui.setAutoShow(false);

		return {stage: cubeCopilot.getStage(facelet)};
	});
	console.log('setup', JSON.stringify(setup));
	if (setup.stage != 'f2l') fail('test position is not F2L stage: ' + setup.stage);

	// do not send any new state: the hint must still appear after the timeout
	await page.waitForTimeout(11000);

	var res = await page.evaluate(function() {
		var overlay = document.querySelector('#copilotHint');
		var leftbar = document.querySelector('#leftbar');
		var cs = overlay ? getComputedStyle(overlay) : null;
		return {
			overlayVisible: !!(cs && cs.display !== 'none' && cs.visibility !== 'hidden' && +cs.opacity > 0),
			overlayDisplay: cs ? cs.display : null,
			overlayText: (overlay ? overlay.textContent : '').replace(/\s+/g, ' ').trim(),
			leftbarOpacity: leftbar ? getComputedStyle(leftbar).opacity : null
		};
	});
	console.log('after idle wait', JSON.stringify(res));

	if (!res.overlayVisible) {
		fail('hint overlay not visible during a solve');
	}
	if (!/F2L/.test(res.overlayText) || !/[URFDLB]/.test(res.overlayText)) {
		fail('hint overlay has no F2L algorithm: ' + res.overlayText);
	}

	// ending the solve must remove the overlay
	var after = await page.evaluate(function() {
		kernel.pushSignal('timerStatus', -1);
		var overlay = document.querySelector('#copilotHint');
		var cs = overlay ? getComputedStyle(overlay) : null;
		return {
			visible: !!(cs && cs.display !== 'none'),
			text: (overlay ? overlay.textContent : '').trim()
		};
	});
	console.log('after solve end', JSON.stringify(after));
	if (after.visible || after.text) {
		fail('hint overlay was not cleared when the solve ended');
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
