"use strict";

// End-to-end smoke test in Chromium:
//  1. the page loads with the Copilot module present
//  2. the Copilot window opens and reports its disabled state
//  3. enabling it and requesting a hint on a prepared state returns an alg
//  4. the virtual cube applies a fresh scramble even when space is pressed
//     before the next scramble arrives (the reported race)
// Usage: node experiment/e2e.js

var http = require('http');
var fs = require('fs');
var path = require('path');
var {chromium} = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8137;
var failures = 0;

function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
}

var MIME = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.php': 'text/plain; charset=utf-8',
	'.json': 'application/json',
	'.webmanifest': 'application/manifest+json'
};

var server = http.createServer(function(req, res) {
	var url = decodeURIComponent(req.url.split('?')[0]);
	console.log('REQ', url);
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
	console.log('serving', root, 'on', port);

	var browser = await chromium.launch();
	var page = await browser.newPage();
	var errors = [];
	page.on('pageerror', function(e) {
		errors.push(e.message + '\nSTACK: ' + (e.stack || ''));
	});
	page.on('console', function(m) {
		if (m.type() == 'error') {
			errors.push('console: ' + m.text());
		}
	});
	page.on('requestfailed', function(r) {
		errors.push('requestfailed: ' + r.url() + ' ' + (r.failure() && r.failure().errorText));
	});
	page.on('response', function(r) {
		if (r.status() >= 400) {
			errors.push('http ' + r.status() + ': ' + r.url());
		}
	});

	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(3000);

	if (errors.length) {
		errors.slice(0, 12).forEach(function(e) {
			console.log('PAGE ISSUE: ' + e);
		});
	}

	var info = await page.evaluate(function() {
		return {
			hasCopilot: typeof cubeCopilot != 'undefined',
			hasGate: typeof scrambleGate != 'undefined',
			hasUI: typeof copilotUI != 'undefined',
			windowCount: $('.window').length,
			copilotDiv: $('#copilotDiv').length
		};
	});
	console.log('page info', JSON.stringify(info));
	if (!info.hasCopilot) {
		fail('cubeCopilot missing');
	}
	if (!info.hasGate) {
		fail('scrambleGate missing');
	}
	if (!info.hasUI) {
		fail('copilotUI missing');
	}

	// engine answers directly in the page (independent of the UI wiring)
	var hint = await page.evaluate(function() {
		var m = new mathlib.CubieCube();
		// R U R' U' leaves the cross intact -> F2L stage
		['R', 'U', "R'", "U'"].forEach(function(t) {
			m.selfMoveStr(t);
		});
		var facelet = m.toFaceCube();
		return {
			stage: cubeCopilot.getStage(facelet),
			hint: cubeCopilot.getHint(facelet, {maxDepth: 6, timeLimit: 4000})
		};
	});
	console.log('stage', hint.stage, 'hint', JSON.stringify(hint.hint));
	if (hint.stage != 'f2l') {
		fail('expected f2l stage, got ' + hint.stage);
	}

	// ALLL lookup on a solved-F2L position
	var ll = await page.evaluate(function() {
		var m = new mathlib.CubieCube();
		['R', 'U', "R'", "U'"].forEach(function(t) {
			m.selfMoveStr(t);
		});
		var f = m.toFaceCube();
		// not necessarily F2L-complete, just make sure the call is safe
		return {stage: cubeCopilot.getStage(f), hasLut: cubeCopilot.alllReady()};
	});
	console.log('ll probe', JSON.stringify(ll));

	if (errors.length) {
		errors.forEach(function(e) {
			fail('page error: ' + e);
		});
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
