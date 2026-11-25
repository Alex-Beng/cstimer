"use strict";

// Layer-by-layer hints through the UI:
//  - the first-layer corner hint highlights the corner and shows a short
//    insertion;
//  - the beginner top-layer steps are labelled instead of OLL/PLL;
//  - an algorithm that passes through cross/F2L states keeps its hint alive.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8195;
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
		return {
			visible: !!(cs && cs.display != 'none'),
			text: (o ? o.textContent : '').replace(/\s+/g, ' ').trim()
		};
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
		kernel.setProp('cpMethod', 'lbl');
		kernel.setProp('cpTimeout', 0);
		kernel.setProp('cpUndoMax', 3);
		kernel.setProp('cpCustom', '');
		tools.getCurPuzzle = function() { return '333'; };
		tools.isCurTrainScramble = function() { return false; };
		cubeCopilot.warmUp();

		function apply(cc, alg) {
			String(alg).split(/\s+/).filter(Boolean).forEach(function(t) {
				cc.selfMoveStr(t);
			});
			return cc;
		}
		function invAlg(alg) {
			return alg.split(/\s+/).filter(Boolean).map(function(t) {
				var m = /^([URFDLB])(2|')?$/.exec(t);
				if (!m) return t;
				if (m[2] == '2') return m[1] + '2';
				if (m[2] == "'") return m[1];
				return m[1] + "'";
			}).reverse().join(' ');
		}

		// a cross-solved position whose first layer is not
		var fl = null;
		var scrambles = [
			'R U F D L B R2 U2 F2 D2 L2 B2',
			'F R U R\' U\' F\' D L2 B2 R2 U2',
			'D2 L2 B2 R2 F2 U2 R U F D L B'
		];
		for (var i = 0; i < scrambles.length && !fl; i++) {
			var cc = apply(new mathlib.CubieCube(), scrambles[i]);
			var cross = cubeCopilot.solveCross(cc.toFaceCube(), {timeLimit: 3000});
			if (!cross) {
				continue;
			}
			apply(cc, cross.join(' '));
			if (cubeCopilot.getStage(cc.toFaceCube(), 'lbl') == 'fl') {
				fl = cc.toFaceCube();
			}
		}

		// the state that F R U R' U' F' solves: F2L done, top edges twisted
		var eoll = invAlg('F R U R\' U\' F\'');
		var eollCc = apply(new mathlib.CubieCube(), eoll);
		var eollFacelet = eollCc.toFaceCube();
		window.__eollFacelet = eollFacelet;
		window.__eollSteps = [];
		var eollHint = cubeCopilot.getHint(eollFacelet, {method: 'lbl', timeLimit: 2000});
		if (eollHint && eollHint.moves) {
			var moves = cubeCopilot.simplifySeq(eollHint.moves);
			var stepCube = new mathlib.CubieCube();
			stepCube.fromFacelet(eollFacelet);
			moves.forEach(function(t) {
				stepCube.selfMoveStr(t);
				window.__eollSteps.push(stepCube.toFaceCube());
			});
		}

		window.__highlightCalls = [];
		var origHighlight = timer.virtual.highlight;
		timer.virtual.highlight = function(slots, color) {
			window.__highlightCalls.push(slots);
			return origHighlight(slots, color);
		};

		return {
			fl: fl,
			flStage: fl ? cubeCopilot.getStage(fl, 'lbl') : null,
			eollStage: cubeCopilot.getStage(eollFacelet, 'lbl'),
			eollHint: eollHint ? {kind: eollHint.kind, alg: eollHint.alg} : null,
			eollSteps: window.__eollSteps.length
		};
	});
	console.log('setup', JSON.stringify(setup));
	if (!setup.fl) fail('no first-layer case could be built');
	if (setup.eollStage != 'eoll') fail('the top-layer setup is not an eoll state: ' + setup.eollStage);
	if (!setup.eollHint || setup.eollHint.kind != 'eoll') fail('no eoll hint for the setup');

	if (setup.fl) {
		await page.evaluate(function(facelet) {
			copilotUI.onFacelet(facelet, facelet);
			copilotUI.requestHint();
		}, setup.fl);
		await page.waitForTimeout(6000);
		var flOverlay = await overlay(page);
		var flHi = await page.evaluate(function() {
			var calls = window.__highlightCalls;
			for (var i = calls.length - 1; i >= 0; i--) {
				if (calls[i]) {
					return calls[i];
				}
			}
			return null;
		});
		console.log('fl overlay', JSON.stringify(flOverlay), 'highlight', JSON.stringify(flHi));
		if (!flOverlay.visible) fail('first-layer hint is not visible');
		if (!/corner|角块/i.test(flOverlay.text)) {
			fail('overlay is not labelled as a first-layer corner: ' + flOverlay.text);
		}
		if (!flHi || flHi.length != 3) {
			fail('first-layer hint does not highlight exactly one corner: ' + JSON.stringify(flHi));
		}
	}

	if (setup.eollHint && setup.eollSteps > 0) {
		await page.evaluate(function() {
			copilotUI.onFacelet(window.__eollFacelet, window.__eollFacelet);
		});
		await page.waitForTimeout(1500);
		var eollOverlay = await overlay(page);
		console.log('eoll overlay', JSON.stringify(eollOverlay));
		if (!eollOverlay.visible) fail('top-layer hint is not visible');
		if (!/Top layer|顶层/i.test(eollOverlay.text)) {
			fail('overlay is not labelled as a beginner top-layer step: ' + eollOverlay.text);
		}

		// the algorithm passes through cross/F2L states; the hint must stay
		var eollSteps = await page.evaluate(function() {
			return window.__eollSteps;
		});
		for (var si = 0; si < eollSteps.length - 1; si++) {
			await page.evaluate(function(st) {
				copilotUI.onFacelet(st, st);
			}, eollSteps[si]);
			await page.waitForTimeout(150);
			var mid = await overlay(page);
			if (!mid.visible) {
				fail('top-layer hint disappeared at step ' + (si + 1));
				break;
			}
		}
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
