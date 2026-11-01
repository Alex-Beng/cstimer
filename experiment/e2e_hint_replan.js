"use strict";

// After a wrong move the hint must re-plan from the current position right
// away (like the smart-cube scramble tracking), instead of disappearing until
// the timeout.  When the wrong move broke the cross, the new hint is a cross
// hint.
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
		// long timeout: an immediate re-plan proves it does not wait for it
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpDrift', 1);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		kernel.setProp('cpF2LCount', 1);
		cubeCopilot.warmUp();

		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var start = cc.toFaceCube();
		var h = cubeCopilot.getHint(start, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 1});

		// a move that is not one of the hinted steps and breaks the cross
		var sigs = cubeCopilot.algStates(start, h.moves);
		var wrong = null;
		['F', 'D', 'L', 'B'].forEach(function(m) {
			if (wrong) return;
			var w = new mathlib.CubieCube();
			w.fromFacelet(start);
			w.selfMoveStr(m);
			var f = w.toFaceCube();
			if (sigs.indexOf(cubeCopilot.normalizeFacelet(f)) < 0) {
				wrong = {move: m, facelet: f};
			}
		});
		if (!wrong) return {error: 'no wrong move found'};

		var rehint = cubeCopilot.getHint(wrong.facelet, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 1});
		window.__wrong = wrong.facelet;
		window.__expectAlg = rehint.moves ? rehint.moves.join(' ') : '';
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [start]);
		return {
			hint: h.alg,
			wrongMove: wrong.move,
			wrongStage: cubeCopilot.getStage(wrong.facelet),
			expectStage: rehint.stage,
			expectAlg: rehint.moves ? rehint.moves.join(' ') : ''
		};
	});
	console.log('setup', JSON.stringify(setup));
	if (setup.error) fail(setup.error);
	if (setup.wrongStage != 'cross') fail('wrong move did not break the cross (test invalid)');
	if (setup.expectStage != 'cross') fail('engine did not produce a cross hint');

	await page.waitForTimeout(6000);
	var first = await overlay(page);
	console.log('first hint', JSON.stringify(first));
	if (!first.visible) fail('no initial hint');

	// wrong move: the hint must re-plan immediately
	await page.evaluate(function() {
		kernel.pushSignal('virtualState', [window.__wrong]);
	});
	await page.waitForTimeout(1500);
	var after = await overlay(page);
	console.log('after wrong move', JSON.stringify(after));
	if (!after.visible) fail('hint disappeared instead of re-planning');
	if (after.text.indexOf(setup.expectAlg) < 0) {
		fail('re-planned hint is not for the current position: ' + after.text);
	}
	if (!/cross/i.test(after.text)) fail('recovery hint is not a cross hint: ' + after.text);

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
