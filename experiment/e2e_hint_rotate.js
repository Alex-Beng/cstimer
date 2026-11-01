"use strict";

// Whole-cube rotations (x/y/z) must not disturb a hint: they arrive as new
// facelets but leave the position unchanged.  Uses cpDrift=1 so any wrong move
// would hide the hint immediately.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8196;
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
		kernel.setProp('cpDrift', 1);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();

		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var start = cc.toFaceCube();
		window.__start = start;
		window.__screen = [start];

		// the on-screen facelet after y and after y y (same position, rotated view)
		var rot = new mathlib.CubieCube();
		rot.fromFacelet(start);
		['y', 'y'].forEach(function(t) {
			rot.selfMoveStr(t);
			window.__screen.push(rot.toFaceCube(null, null, null, true));
		});

		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [start]);
		return {
			start: start,
			sameNorm: [0, 1, 2].map(function(i) {
				return cubeCopilot.canonicalSig(window.__screen[i]) == cubeCopilot.canonicalSig(start);
			}),
			distinctRaw: window.__screen[1] != start && window.__screen[2] != start
		};
	});
	console.log('setup', JSON.stringify(setup));
	if (!setup.distinctRaw) fail('rotation did not produce a new facelet (test invalid)');
	if (setup.sameNorm.indexOf(false) >= 0) fail('rotation changed the normalised position (test invalid)');

	await page.waitForTimeout(1800);
	var shown = await overlay(page);
	console.log('initial', JSON.stringify(shown));
	if (!shown.visible) fail('no hint before rotating');

	// two rotations of the whole cube, as the virtual cube reports them
	for (var i = 1; i <= 2; i++) {
		await page.evaluate(function(i) {
			kernel.pushSignal('virtualState', [window.__screen[i]]);
		}, i);
		await page.waitForTimeout(150);
		var now = await overlay(page);
		console.log('after rotation', i, JSON.stringify(now));
		if (!now.visible) {
			fail('hint disappeared after whole-cube rotation ' + i);
			break;
		}
	}

	// and following the hint still works afterwards (the view is back to the
	// original orientation, then the first move of the hint is applied)
	var follow = await page.evaluate(function() {
		kernel.pushSignal('virtualState', [window.__start]);
		var h = cubeCopilot.getHint(window.__start, {maxDepth: 8, timeLimit: 4000, llMode: 'op'});
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(window.__start);
		cc.selfMoveStr(h.moves[0]);
		kernel.pushSignal('virtualState', [cc.toFaceCube()]);
		return h.moves[0];
	});
	await page.waitForTimeout(150);
	var after = await overlay(page);
	console.log('after following', follow, JSON.stringify(after));
	if (!after.visible) fail('hint disappeared when following after a rotation');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
