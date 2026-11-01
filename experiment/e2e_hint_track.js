"use strict";

// Hint lifetime + custom algorithm overrides.
//
// 1. a shown hint must survive moves that follow it (showing the remaining alg)
// 2. wrong moves are answered with the inverse moves back to the algorithm
// 3. cpTimeout must accept 1s
// 4. the custom-alg editor must save a valid alg and reject an invalid one
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8193;
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

	// ---------------------------------------------------------------- setup
	var setup = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 1);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();

		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var facelet = cc.toFaceCube();
		window.__S = facelet;
		var hint = cubeCopilot.getHint(facelet, {maxDepth: 8, timeLimit: 4000, llMode: 'op'});
		window.__hint = hint;
		window.__mm = [];
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [facelet]);
		return {alg: hint.alg, moves: hint.moves, key: hint.key, stage: hint.stage};
	});
	console.log('hint', JSON.stringify(setup));
	if (!setup.moves || setup.moves.length < 2) {
		fail('test hint is too short: ' + setup.alg);
	}

	await page.waitForTimeout(6000);
	var before = await overlay(page);
	console.log('idle hint', JSON.stringify(before));
	if (!before.visible) {
		fail('hint did not appear with cpTimeout=1');
	}

	// ------------------------------------------- follow the hinted moves
	var follow = await page.evaluate(function() {
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(window.__S);
		cc.selfMoveStr(window.__hint.moves[0]);
		var next = cc.toFaceCube();
		kernel.pushSignal('virtualState', [next]);
		window.__cur = next;
		return {remaining: window.__hint.moves.slice(1).join(' ')};
	});
	var following = await overlay(page);
	console.log('after following move', JSON.stringify(following), 'expect remaining:', follow.remaining);
	if (!following.visible) {
		fail('hint disappeared after a move that follows it');
	}
	if (follow.remaining && following.text.indexOf(follow.remaining) < 0) {
		fail('overlay does not show the remaining algorithm');
	}

	// ------------------------------------------- unrelated moves + drift
	var drift = await page.evaluate(function() {
		var sigs = cubeCopilot.algStates(window.__S, window.__hint.moves);
		var cur = window.__cur || window.__S;
		var pushed = [];
		['F', 'D', 'L', 'B', 'U2'].forEach(function(m) {
			if (pushed.length >= 2) return;
			var cc = new mathlib.CubieCube();
			cc.fromFacelet(cur);
			cc.selfMoveStr(m);
			var next = cc.toFaceCube();
			if (sigs.indexOf(cubeCopilot.normalizeFacelet(next)) < 0) {
				cur = next;
				pushed.push(next);
				kernel.pushSignal('virtualState', [next]);
			}
		});
		var last = pushed[pushed.length - 1];
		return {
			pushed: pushed.length,
			last: last
		};
	});
	console.log('drift pushes', JSON.stringify(drift));
	if (drift.pushed < 2) {
		fail('could not build two non-matching states');
	}
	// wrong moves are answered with the inverse sequence that leads back
	await page.waitForTimeout(400);
	var afterDrift = await overlay(page);
	console.log('after drift', JSON.stringify(afterDrift));
	if (!afterDrift.visible) {
		fail('hint disappeared instead of showing the undo moves');
	} else if (afterDrift.text.indexOf("U' R'") < 0) {
		fail('undo is not part of the formula: ' + afterDrift.text);
	}
	if (/undo|回退/i.test(afterDrift.text)) fail('separate undo prompt still shown');

	// ------------------------------------------- custom algorithm editor
	var editor = await page.evaluate(function() {
		kernel.pushSignal('timerStatus', -1);
		kernel.setProp('cpCustom', '');
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [window.__S]);
		return true;
	});

	// wait for a fresh hint
	await page.waitForTimeout(6000);
	var hint2 = await overlay(page);
	console.log('second hint', JSON.stringify(hint2));

	var save = await page.evaluate(function() {
		var div = document.querySelector('#copilotDiv');
		var input = div.querySelector('input[type=text]');
		var buttons = [].slice.call(div.querySelectorAll('input[type=button]'));
		var saveBtn = buttons.filter(function(b) { return /save/i.test(b.value); })[0];
		if (!input || !saveBtn) {
			return {error: 'editor controls not found', hasInput: !!input, buttons: buttons.map(function(b) { return b.value; })};
		}
		var hint = window.__hint;
		input.value = hint.alg;
		saveBtn.click();
		return {custom: kernel.getProp('cpCustom'), key: hint.key};
	});
	console.log('save valid', JSON.stringify(save));
	if (save.error) {
		fail(save.error);
	} else {
		var map = {};
		try { map = JSON.parse(save.custom || '{}'); } catch (e) {}
		if (!map[save.key]) {
			fail('valid algorithm was not saved under ' + save.key);
		}
	}
	var afterSave = await overlay(page);
	if (!afterSave.visible) {
		fail('hint disappeared after saving a custom algorithm');
	}

	var bad = await page.evaluate(function() {
		var div = document.querySelector('#copilotDiv');
		var input = div.querySelector('input[type=text]');
		var buttons = [].slice.call(div.querySelectorAll('input[type=button]'));
		var saveBtn = buttons.filter(function(b) { return /save/i.test(b.value); })[0];
		var before = kernel.getProp('cpCustom');
		input.value = 'R U';
		saveBtn.click();
		return {before: before, after: kernel.getProp('cpCustom')};
	});
	console.log('save invalid', JSON.stringify(bad));
	if (bad.before != bad.after) {
		fail('an invalid algorithm was saved');
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
