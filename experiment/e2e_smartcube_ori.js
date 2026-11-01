"use strict";

// Smart-cube frame: csTimer's "Cube orientation" setting must also orient the
// hint, so the algorithm uses the face names the user sees in their grip.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8197;
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

function overlay(page) {
	return page.evaluate(function() {
		var o = document.querySelector('#copilotHint');
		var cs = o ? getComputedStyle(o) : null;
		return {visible: !!(cs && cs.display != 'none'), text: (o ? o.textContent : '').replace(/\s+/g, ' ').trim()};
	});
}

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
		kernel.setProp('cpTimeout', 0);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();
		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var raw = cc.toFaceCube();
		window.__raw = raw;

		var ori = 3; // "(UR) y" in the Cube orientation list
		kernel.setProp('giiOri', String(ori));
		var plain = cubeCopilot.getHint(raw, {maxDepth: 8, timeLimit: 1500, llMode: 'op'});
		var orientedFacelet = cubeCopilot.orientFacelet(raw, ori);
		var oriented = cubeCopilot.getHint(orientedFacelet, {maxDepth: 8, timeLimit: 1500, llMode: 'op'});
		window.__plainAlg = plain.alg;
		window.__oriAlg = oriented.alg;

		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('giikerState', [raw]);
		return {plain: plain.alg, oriented: oriented.alg, raw: raw};
	});
	console.log('setup', JSON.stringify(setup));
	if (!setup.plain || !setup.oriented) fail('no hint for the test state');
	if (setup.plain == setup.oriented) fail('orientation did not change the hint (test invalid)');

	await page.waitForTimeout(1500);
	var shown = await overlay(page);
	console.log('giiOri=3 overlay', JSON.stringify(shown));
	if (!shown.visible) fail('no hint for a smart-cube state');
	if (shown.text.indexOf(setup.oriented) < 0) {
		fail('hint is not expressed in the chosen cube orientation');
	}
	if (shown.text.indexOf(setup.plain) >= 0) {
		fail('hint still uses the raw frame');
	}

	// auto must fall back to the raw frame
	await page.evaluate(function() {
		kernel.setProp('giiOri', 'auto');
		kernel.pushSignal('giikerState', [window.__raw]);
	});
	await page.waitForTimeout(1500);
	var plainShown = await overlay(page);
	console.log('giiOri=auto overlay', JSON.stringify(plainShown));
	if (!plainShown.visible) fail('no hint after switching back to auto');
	if (plainShown.text.indexOf(setup.plain) < 0) {
		fail('auto orientation does not use the raw frame');
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
