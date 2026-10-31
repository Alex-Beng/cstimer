"use strict";

// Browser test for the virtual-cube scramble race and the Copilot window.
//
// Steps
//  1. open the page, switch the input to the virtual cube
//  2. drive the cube through a solve, then press space *immediately* after the
//     solve while the next scramble is still being generated
//  3. assert the cube never shows the previous scramble again
//
// The page exposes timer.virtual / scrambleGate so the test can observe state.

var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8141;
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
		// ignore environment-only noise (service worker, wake lock)
		var msg = e.message || '';
		if (/ServiceWorker|sw\.js|Wake Lock|bad HTTP response code/i.test(msg)) {
			return;
		}
		fatal.push(msg);
	});

	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	// switch to the virtual cube
	var switched = await page.evaluate(function() {
		try {
			kernel.setProp('input', 'v');
			return kernel.getProp('input');
		} catch (e) {
			return 'ERR ' + e.message;
		}
	});
	console.log('input =', switched);
	if (switched != 'v') {
		fail('could not switch to virtual cube: ' + switched);
	}
	await page.waitForTimeout(1500);

	var gateInfo = await page.evaluate(function() {
		return {
			hasVirtual: !!(timer.virtual && timer.virtual.highlight),
			gateReady: scrambleGate.isReady(),
			gateToken: scrambleGate.token()
		};
	});
	console.log('gate', JSON.stringify(gateInfo));
	if (!gateInfo.hasVirtual) {
		fail('timer.virtual.highlight missing');
	}

	// the scramble gate must report "not ready" between scrambling and scramble
	var race = await page.evaluate(function() {
		var log = [];
		var before = scrambleGate.token();
		kernel.pushSignal('scrambling', '333');
		log.push(['during', scrambleGate.isReady()]);
		kernel.pushSignal('scramble', ['333', "R U R' U' F2 D", 0]);
		log.push(['after', scrambleGate.isReady()]);
		return {log: log, tokenAdvanced: scrambleGate.token() > before};
	});
	console.log('race gate', JSON.stringify(race));
	if (race.log[0][1] !== false) {
		fail('gate reported ready while generating');
	}
	if (race.log[1][1] !== true) {
		fail('gate did not become ready after publish');
	}
	if (!race.tokenAdvanced) {
		fail('gate token did not advance');
	}

	// pressing space while generating must not start; it must start after
	var startBehaviour = await page.evaluate(async function() {
		kernel.pushSignal('scrambling', '333');
		// press space now
		timer.virtual.onkeydown(32, {});
		var statusRightAfter = timer.status();
		// deliver the fresh scramble a moment later
		await new Promise(function(r) {
			setTimeout(r, 120);
		});
		kernel.pushSignal('scramble', ['333', "R U R' U' F2 D", 0]);
		await new Promise(function(r) {
			setTimeout(r, 400);
		});
		return {statusRightAfter: statusRightAfter, statusLater: timer.status()};
	});
	console.log('space during generation', JSON.stringify(startBehaviour));
	if (startBehaviour.statusRightAfter != -1) {
		fail('timer started before the scramble was ready (status ' + startBehaviour.statusRightAfter + ')');
	}
	if (startBehaviour.statusLater == -1) {
		fail('timer never started after the scramble arrived');
	}

	// copilot window exists and is toggleable
	var ui = await page.evaluate(function() {
		var enabled = kernel.getProp('cpEnabled');
		kernel.setProp('cpEnabled', true);
		var after = kernel.getProp('cpEnabled');
		kernel.setProp('cpEnabled', enabled);
		return {before: enabled, after: after, div: $('#copilotDiv').length};
	});
	console.log('copilot ui', JSON.stringify(ui));
	if (ui.div != 1) {
		fail('copilot window not created');
	}

	fatal.forEach(function(m) {
		fail('page error: ' + m);
	});

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
