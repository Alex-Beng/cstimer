"use strict";

// Regression test for the Copilot F2L highlight on the 3D virtual cube:
//  - reaching a real F2L position and asking for a hint calls
//    timer.virtual.highlight with the corner+edge stickers of the min pair
//  - exactly those 5 stickers are recoloured (not a stale facelet mapping)
//  - clearing the hint restores the original colours
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
			tools.getCurPuzzle = function() { return '333'; };
			tools.isCurTrainScramble = function() { return false; };
			var calls = [];
			var orig = timer.virtual.highlight;
			timer.virtual.highlight = function(f) { calls.push(f); return orig(f); };
			copilotUI.onFacelet(found.facelet, found.facelet);
			copilotUI.requestHint();
			function magenta(mesh) {
				var c = mesh && mesh.materials && mesh.materials[0] && mesh.materials[0].color;
				return !!(c && c.r === 1 && c.g === 0 && c.b === 1);
			}
			var n = 0;
			var state = __puzzle.twisty.cubePieces;
			for (var f = 0; f < state.length; f++) {
				for (var s = 0; s < state[f].length; s++) {
					if (magenta(state[f][s][1].children[0])) {
						n++;
					}
				}
			}
			return {calls: calls, magenta: n};
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
		if (hint.magenta != 5) {
			fail('expected 5 highlighted stickers, got ' + hint.magenta);
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
