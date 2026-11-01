"use strict";

// A half turn in the hint (R2) must accept the half-way state (R) as progress.
// Uses cpDrift=1 so any non-matching move would hide the hint immediately.
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
		kernel.setProp('cpTimeout', 1);
		kernel.setProp('cpDrift', 1);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();

		function inv(tokens) {
			return tokens.slice().reverse().map(function(t) {
				var m = /^([URFDLB]w?|[EMSxyz])(2|')?$/.exec(t);
				if (!m) return t;
				if (m[2] == '2') return m[1] + '2';
				if (m[2] == "'") return m[1];
				return m[1] + "'";
			});
		}
		function stateFor(alg) {
			var cc = new mathlib.CubieCube();
			inv(alg.split(/\s+/).filter(Boolean)).forEach(function(t) { cc.selfMoveStr(t); });
			return cc.toFaceCube();
		}

		// OLL 5 starts with a wide move and then a half turn
		var start = stateFor("Rw' U2 R U R' U Rw");
		var hint = cubeCopilot.getHint(start, {maxDepth: 8, timeLimit: 4000, llMode: 'op'});

		// every facelet the cube will report while following the hint, with the
		// quarter turn of each half turn inserted
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(start);
		var pushes = [];
		var halfIndex = -1;
		for (var i = 0; i < hint.moves.length; i++) {
			var t = hint.moves[i];
			if (/2$/.test(t)) {
				var q = new mathlib.CubieCube();
				q.init(cc.ca.slice(), cc.ea.slice());
				q.ori = cc.ori;
				q.selfMoveStr(t.slice(0, -1));
				pushes.push({facelet: q.toFaceCube(), label: t + ' half', index: i});
				if (halfIndex < 0) halfIndex = pushes.length - 1;
			}
			cc.selfMoveStr(t);
			pushes.push({facelet: cc.toFaceCube(), label: t, index: i});
		}
		window.__start = start;
		window.__pushes = pushes;
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [start]);
		return {alg: hint.alg, moves: hint.moves, halfIndex: halfIndex, pushes: pushes.length};
	});
	console.log('setup', JSON.stringify({alg: setup.alg, moves: setup.moves, halfIndex: setup.halfIndex}));
	if (setup.halfIndex < 0) fail('test hint has no half turn');

	await page.waitForTimeout(6000);
	var shown = await overlay(page);
	console.log('initial overlay', JSON.stringify(shown));
	if (!shown.visible) fail('no hint shown before following');

	// follow the hint: push each state, stopping at the half-way state and
	// checking that the hint is still there
	for (var k = 0; k <= setup.halfIndex; k++) {
		var p = await page.evaluate(function(k) {
			var pi = window.__pushes[k];
			kernel.pushSignal('virtualState', [pi.facelet]);
			return pi;
		}, k);
		await page.waitForTimeout(150);
		var now = await overlay(page);
		console.log('push', k, p.label, '->', JSON.stringify(now));
		if (!now.visible) {
			fail('hint disappeared after "' + p.label + '"');
			break;
		}
		if (k == setup.halfIndex) {
			var quarter = p.label.split(/\s+/)[0].replace(/2$/, '');
			if (now.text.indexOf(quarter) < 0) {
				fail('remaining algorithm does not show the second quarter turn: ' + now.text);
			}
		}
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
