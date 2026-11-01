"use strict";

// Tracking the hinted algorithm: a wrong move must be answered with the
// inverse move(s) that return to the correct point of the algorithm (like the
// smart-cube scramble tracking re-syncing after a wrong move).
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8198;
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
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		kernel.setProp('cpF2LCount', 1);
		cubeCopilot.warmUp();

		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var start = cc.toFaceCube();
		var h = cubeCopilot.getHint(start, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 1});
		var sigs = cubeCopilot.algStates(start, h.moves);

		function step(facelet, token) {
			var c = new mathlib.CubieCube();
			c.fromFacelet(facelet);
			c.selfMoveStr(token);
			return c.toFaceCube();
		}
		function pick(facelet, list) {
			for (var i = 0; i < list.length; i++) {
				var f = step(facelet, list[i]);
				if (sigs.indexOf(cubeCopilot.normalizeFacelet(f)) < 0) {
					return {move: list[i], facelet: f};
				}
			}
			return null;
		}
		var w1 = pick(start, ['F', 'D', 'L', 'B', 'U', 'R']);
		var w2 = w1 && pick(w1.facelet, ['R', 'U', 'L', 'B', 'F', 'D']);
		if (!w1 || !w2) return {error: 'could not build two wrong moves'};

		window.__start = start;
		window.__w1 = w1.facelet;
		window.__w2 = w2.facelet;
		window.__u1 = cubeCopilot.invertToken(w1.move);
		window.__u2 = cubeCopilot.invertToken(w2.move);
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [start]);
		return {
			hint: h.alg,
			m1: w1.move,
			m2: w2.move,
			u1: window.__u1,
			u2: window.__u2
		};
	});
	console.log('setup', JSON.stringify(setup));
	if (setup.error) fail(setup.error);

	await page.waitForTimeout(6000);
	var first = await overlay(page);
	console.log('first', JSON.stringify(first));
	if (!first.visible || first.text.indexOf(setup.hint) < 0) fail('no initial hint');

	// one wrong move -> its inverse
	await page.evaluate(function() {
		kernel.pushSignal('virtualState', [window.__w1]);
	});
	await page.waitForTimeout(300);
	var one = await overlay(page);
	console.log('one wrong', JSON.stringify(one));
	if (!one.visible) fail('hint disappeared on a wrong move');
	if (one.text.indexOf(setup.u1) < 0) fail('undo move missing: ' + one.text);
	if (!/undo|回退/i.test(one.text)) fail('no undo label: ' + one.text);

	// a second wrong move -> newest undo first
	await page.evaluate(function() {
		kernel.pushSignal('virtualState', [window.__w2]);
	});
	await page.waitForTimeout(300);
	var two = await overlay(page);
	console.log('two wrong', JSON.stringify(two));
	if (!two.visible) fail('hint disappeared on the second wrong move');
	if (two.text.indexOf(setup.u2 + ' ' + setup.u1) < 0) fail('undo stack out of order: ' + two.text);

	// following the first undo removes it from the list
	await page.evaluate(function() {
		var c = new mathlib.CubieCube();
		c.fromFacelet(window.__w2);
		c.selfMoveStr(window.__u2);
		kernel.pushSignal('virtualState', [c.toFaceCube()]);
	});
	await page.waitForTimeout(300);
	var pop = await overlay(page);
	console.log('after first undo', JSON.stringify(pop));
	if (!pop.visible) fail('hint disappeared while undoing');
	if (pop.text.indexOf(setup.u1) < 0) fail('remaining undo missing: ' + pop.text);
	if (setup.u1 != setup.u2 && pop.text.indexOf(setup.u2) >= 0) fail('undo move was not consumed');

	// following the second undo returns to the algorithm
	await page.evaluate(function() {
		var c = new mathlib.CubieCube();
		c.fromFacelet(window.__w1);
		c.selfMoveStr(window.__u1);
		kernel.pushSignal('virtualState', [c.toFaceCube()]);
	});
	await page.waitForTimeout(500);
	var back = await overlay(page);
	console.log('back on track', JSON.stringify(back));
	if (!back.visible) fail('hint disappeared after undoing the deviation');
	if (back.text.indexOf(setup.hint) < 0) fail('hint did not resume: ' + back.text);
	if (/undo|回退/i.test(back.text)) fail('still showing the undo after returning');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
