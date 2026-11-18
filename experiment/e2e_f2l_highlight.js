"use strict";

// Regression test for the Copilot F2L highlight on the 3D virtual cube:
//  - reaching a real F2L position and asking for a hint calls
//    timer.virtual.highlight with the corner+edge stickers of the min pair
//  - exactly those 5 stickers get a highlighted border (not a stale mapping)
//  - the sticker fill colour is kept
// Usage: node experiment/e2e_f2l_highlight.js
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8203;
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
	var page = await browser.newPage({viewport: {width: 1200, height: 900}});
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	// capture the puzzle object and switch to the 3D virtual cube
	await page.evaluate(function() {
		var orig = puzzleFactory.init;
		puzzleFactory.init = function(options, ml, parent, cb) {
			return orig(options, ml, parent, function(ret, isInit) {
				if (ret) window.__puzzle = ret;
				cb(ret, isInit);
			});
		};
		kernel.setProp('vrcSpeed', 0);
		kernel.setProp('input', 'v');
	});
	await page.waitForTimeout(3000);

	var setup = await page.evaluate(function() {
		function apply(str) {
			__puzzle.applyMoves(__puzzle.parseScramble(str, true));
		}
		if (!window.__puzzle) {
			return {err: 'no virtual cube'};
		}
		// normalise the random display state to solved, then build an F2L case
		var sol = min2phase.solve(__puzzle.getFacelet());
		if (sol && !/^Error/.test(sol)) {
			apply(sol);
		}
		var found = null;
		for (var t = 0; t < 6 && !found; t++) {
			apply('R U F D L B R2 U2 F2 D2 L2 B2 R U F');
			var cross = cubeCopilot.solveCross(__puzzle.getFacelet(), {maxDepth: 8});
			if (!cross) {
				continue;
			}
			apply(cross.join(' '));
			var f = __puzzle.getFacelet();
			var pair = cubeCopilot.minPair(f);
			if (cubeCopilot.getStage(f) == 'f2l' && pair && !pair.solved) {
				found = {facelet: f, pieces: pair.pieces};
			} else {
				apply('U R2 F2 D');
			}
		}
		return {found: found, err: found ? null : 'no F2L case'};
	});
	console.log('setup', JSON.stringify(setup));
	if (!setup.found) {
		fail(setup.err);
	}

	if (setup.found) {
		var hint = await page.evaluate(function(found) {
			kernel.setProp('cpEnabled', true);
			kernel.setProp('vrcMP', 'cfop');
			tools.getCurPuzzle = function() { return '333'; };
			tools.isCurTrainScramble = function() { return false; };
			var calls = [];
			var orig = timer.virtual.highlight;
			timer.virtual.highlight = function(f) { calls.push(f); return orig(f); };
			copilotUI.onFacelet(found.facelet, found.facelet);
			copilotUI.requestHint();
			function gold(mat) {
				var c = mat && mat.color;
				return !!(mat && mat.wireframe && c && c.r === 1 && c.g > 0.6 && c.b < 0.8);
			}
			var n = 0;
			var fillOverwritten = 0;
			var state = __puzzle.twisty.cubePieces;
			for (var f = 0; f < state.length; f++) {
				for (var s = 0; s < state[f].length; s++) {
					var mats = state[f][s][1].children[0].materials;
					if (mats && gold(mats[1])) {
						n++;
						if (mats[0] && mats[0].wireframe) {
							fillOverwritten++;
						}
					}
				}
			}
			return {calls: calls, gold: n, fillOverwritten: fillOverwritten};
		}, setup.found);
		console.log('hint', JSON.stringify(hint));
		var call = null;
		for (var i = hint.calls.length - 1; i >= 0; i--) {
			if (hint.calls[i]) {
				call = hint.calls[i];
				break;
			}
		}
		if (!call || call.join(',') != setup.found.pieces.join(',')) {
			fail('highlighted ' + JSON.stringify(call) + ', expected ' + JSON.stringify(setup.found.pieces));
		}
		if (hint.gold != 5) {
			fail('expected 5 gold highlighted borders, got ' + hint.gold);
		}
		if (hint.fillOverwritten) {
			fail('sticker fill colour was overwritten');
		}

		var customColor = await page.evaluate(function(found) {
			kernel.setProp('cpColor', '#ff00ff');
			var passed = null;
			var orig = timer.virtual.highlight;
			timer.virtual.highlight = function(slots, color) {
				passed = color;
				return orig(slots, color);
			};
			copilotUI.onFacelet(found.facelet, found.facelet);
			copilotUI.requestHint();
			timer.virtual.highlight = orig;
			__puzzle.highlight(found.pieces, 0xff00ff);
			var magenta = 0;
			var state = __puzzle.twisty.cubePieces;
			for (var f = 0; f < state.length; f++) {
				for (var s2 = 0; s2 < state[f].length; s2++) {
					var m = state[f][s2][1].children[0].materials[1];
					var c = m && m.color;
					if (m && m.wireframe && c && c.r === 1 && c.g === 0 && c.b === 1) {
						magenta++;
					}
				}
			}
			kernel.setProp('cpColor', '#ffd700');
			__puzzle.highlight(found.pieces, 0xffd700);
			return {passed: passed, magenta: magenta};
		}, setup.found);
		console.log('custom color', JSON.stringify(customColor));
		if (customColor.passed !== 0xff00ff) {
			fail('copilot did not pass the custom highlight color');
		}
		if (customColor.magenta != 5) {
			fail('expected 5 magenta highlighted borders, got ' + customColor.magenta);
		}




		var turned = await page.evaluate(function(found) {
			kernel.setProp('cpEnabled', true);
			kernel.setProp('vrcMP', 'cfop');
			var turned = cubeCopilot.orientFacelet(found.facelet, 3);
			var calls = [];
			var orig = timer.virtual.highlight;
			timer.virtual.highlight = function(f) { calls.push(f); return orig(f); };
			copilotUI.onFacelet(turned, turned);
			timer.virtual.highlight = orig;
			var after = null;
			for (var i = calls.length - 1; i >= 0; i--) {
				if (calls[i]) {
					after = calls[i];
					break;
				}
			}
			function colors(fl, pieces) {
				return pieces.map(function(idx) { return fl[idx]; }).sort().join('');
			}
			return {
				before: colors(found.facelet, found.pieces),
				after: after ? colors(turned, after) : null
			};
		}, setup.found);
		console.log('rotation', JSON.stringify(turned));
		if (turned.before != turned.after) {
			fail('turning the cube changed the highlighted pair');
		}

		var gate = await page.evaluate(function(found) {
			function probe(timerMethod, hintMethod) {
				kernel.setProp('cpMethod', hintMethod);
				kernel.setProp('vrcMP', timerMethod);
				kernel.pushSignal('virtualState', [found.facelet]);
				var n = 0;
				var orig = timer.virtual.highlight;
				timer.virtual.highlight = function(f) { n++; return orig(f); };
				copilotUI.requestHint();
				timer.virtual.highlight = orig;
				return n;
			}
			return {
				off: probe('cfop', 'off'),
				withN: probe('n', 'cfop'),
				withRoux: probe('roux', 'cfop'),
				withEdge: probe('edge', 'cfop')
			};
		}, setup.found);
		console.log('gating', JSON.stringify(gate));
		if (gate.off) {
			fail('copilot hinted while its hint method was off');
		}
		if (!gate.withN || !gate.withRoux || !gate.withEdge) {
			fail('copilot hints still depend on the timer method');
		}


		function sampleFlash() {
			return page.evaluate(function() {
				var state = __puzzle.twisty.cubePieces;
				for (var f = 0; f < state.length; f++) {
					for (var s = 0; s < state[f].length; s++) {
						var m = state[f][s][1].children[0].materials[1];
						var c = m && m.color;
						if (m && m.wireframe && c && c.r === 1 && c.g > 0.6 && c.b < 0.8) {
							return [c.r, c.g, c.b, m.opacity, m.wireframeLinewidth, !!m.transparent];
						}
					}
				}
				return null;
			});
		}
		var flash1 = await sampleFlash();
		await page.waitForTimeout(350);
		var flash2 = await sampleFlash();
		if (!flash1 || !flash2 || flash1.join(',') == flash2.join(',')) {
			fail('highlight does not flash');
		}
		if (!flash1 || !flash2 || !flash1[5] || !flash2[5]) {
			fail('highlight border is not transparent, the flash cannot render');
		}

		var rect = await page.evaluate(function() {
			var cv = document.querySelector('#container canvas');
			var r = cv.getBoundingClientRect();
			return {x: r.x, y: r.y, width: r.width, height: r.height};
		});
		var on = await page.screenshot({clip: rect});
		await page.evaluate(function() { timer.virtual.highlight(null); });
		await page.waitForTimeout(400);
		var off = await page.screenshot({clip: rect});
		if (Buffer.compare(on, off) === 0) {
			fail('highlight did not change the rendered cube');
		}
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})().catch(function(e) {
	console.error(e);
	process.exit(1);
});
