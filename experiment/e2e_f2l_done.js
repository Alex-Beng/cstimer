"use strict";

// 1. A completed F2L pair must disappear from a multi-pair hint (no leftover
//    last move).
// 2. Rotating the virtual cube must re-express the hint in the new frame.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8199;
var failures = 0;
function fail(m) { failures++; console.log('FAIL ' + m); }

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

var MULTI = 'FUULUUULRUFFURUBRFLFBBFBFFRDDDDDDDDDLBBRLRLLLRRULBFRBB';

(async function() {
	await new Promise(function(r) { server.listen(port, '0.0.0.0', r); });
	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var setup = await page.evaluate(function(multi) {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		kernel.setProp('cpF2LCount', 2);
		cubeCopilot.warmUp();
		var h = cubeCopilot.getHint(multi, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 2});
		var first = h.pairs[0];
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(multi);
		var states = [];
		first.moves.forEach(function(t) {
			cc.selfMoveStr(t);
			states.push(cc.toFaceCube());
		});
		var finalFacelet = cc.toFaceCube();
		var next = cubeCopilot.getHint(finalFacelet, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 2});
		window.__states = states;
		window.__final = finalFacelet;
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [multi]);
		return {
			pair: first.name,
			alg: first.alg,
			nextAlg: next && next.moves ? next.moves.join(' ') : ''
		};
	}, MULTI);
	console.log('setup', JSON.stringify(setup));

	await page.waitForTimeout(6000);
	var before = await overlay(page);
	console.log('before', JSON.stringify(before));
	if (!before.visible) fail('no initial multi-pair hint');

	// follow the first pair's algorithm step by step
	await page.evaluate(function() {
		window.__states.forEach(function(f) {
			kernel.pushSignal('virtualState', [f]);
		});
	});
	await page.waitForTimeout(1600);
	var after = await overlay(page);
	console.log('after completing pair', JSON.stringify(after));
	if (!after.visible) fail('no hint after completing the pair');
	if (after.text.indexOf(setup.pair + ':') >= 0) {
		fail('completed pair is still listed: ' + after.text);
	}
	if (setup.nextAlg && after.text.indexOf(setup.nextAlg) < 0) {
		fail('hint did not move on to the new position: ' + after.text);
	}

	// rotating the virtual cube re-expresses the hint in the new frame
	var rot = await page.evaluate(function() {
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(window.__final);
		cc.selfMoveStr('y');
		var f = cc.toFaceCube(null, null, null, true);
		var h = cubeCopilot.getHint(f, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 2});
		window.__rotated = f;
		window.__rotAlg = h && h.moves ? h.moves.join(' ') : '';
		kernel.pushSignal('virtualState', [f]);
		return {alg: window.__rotAlg};
	});
	await page.waitForTimeout(1600);
	var afterRot = await overlay(page);
	console.log('after rotation', JSON.stringify(afterRot), 'expected', JSON.stringify(rot));
	if (!afterRot.visible) fail('hint disappeared after rotating the virtual cube');
	if (rot.alg && afterRot.text.indexOf(rot.alg) < 0) {
		fail('hint was not re-expressed in the rotated frame: ' + afterRot.text);
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
