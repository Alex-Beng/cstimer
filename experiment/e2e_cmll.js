"use strict";

// Roux through the UI: once both blocks are solved the copilot must show a
// CMLL algorithm, and the hint must disappear after the corners are solved.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8204;
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

	var caseState = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpMethod', 'roux');
		kernel.setProp('cpTimeout', 0);
		kernel.setProp('cpCustom', '');
		cubeCopilot.ensureCmllLut();

		function inv(tokens) {
			return tokens.slice().reverse().map(function(t) {
				var m = /^([URFDLB]w?|[EMS])(2|'|2')?$/.exec(t);
				if (!m) return t;
				if (m[2] == '2') return m[1] + '2';
				if (m[2] == "'") return m[1];
				return m[1] + "'";
			});
		}
		for (var i = 0; i < CMLL_DATA.length; i++) {
			var alg = CMLL_DATA[i].split('|')[1];
			if (!alg || /[xyzXYZ]/.test(alg)) continue;
			var cc = new mathlib.CubieCube();
			inv(alg.split(/\s+/).filter(Boolean)).forEach(function(t) {
				cc.selfMoveStr(t);
			});
			var facelet = cc.toFaceCube();
			if (cubeCopilot.getStage(facelet, 'roux') != 'cmll') continue;
			var hint = cubeCopilot.getHint(facelet, {method: 'roux'});
			if (hint && hint.kind == 'cmll') {
				var after = new mathlib.CubieCube();
				after.fromFacelet(facelet);
				hint.moves.forEach(function(t) {
					after.selfMoveStr(t);
				});
				return {
					facelet: facelet,
					name: hint.name,
					alg: hint.alg,
					moves: hint.moves,
					afterFacelet: after.toFaceCube(),
					afterStage: cubeCopilot.getStage(after.toFaceCube(), 'roux')
				};
			}
		}
		return null;
	});
	console.log('case', JSON.stringify(caseState));
	if (!caseState) fail('no CMLL case found');
	if (caseState && caseState.afterStage == 'cmll') fail('CMLL case is not solved by its own algorithm');

	await page.evaluate(function(st) {
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [st.facelet]);
	}, caseState);
	await page.waitForTimeout(2500);
	var shown = await overlay(page);
	console.log('cmll overlay', JSON.stringify(shown));
	if (!shown.visible) fail('CMLL hint not shown');
	else if (!/CMLL/.test(shown.text)) fail('overlay does not show a CMLL hint: ' + shown.text);
	if (caseState && shown.text.indexOf(caseState.alg) == -1) {
		fail('overlay does not show the expected algorithm: ' + shown.text);
	}

	await page.evaluate(function(st) {
		kernel.pushSignal('virtualState', [st.afterFacelet]);
	}, caseState);
	await page.waitForTimeout(1000);
	var after = await overlay(page);
	console.log('after cmll', JSON.stringify(after));
	if (after.visible && /CMLL/.test(after.text)) fail('CMLL hint did not clear after the algorithm');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
