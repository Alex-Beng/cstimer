"use strict";

// Reaching the last layer (CF done) must hint immediately, even with a long
// cpTimeout: it is a stage milestone, not a "no progress" timeout.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8200;
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

(async function() {
	await new Promise(function(r) { server.listen(port, '0.0.0.0', r); });
	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var info = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 30);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();
		function inv(tokens) {
			return tokens.slice().reverse().map(function(x) {
				var m = /^([URFDLB]w?|[EMSxyz])(2|')?$/.exec(x);
				if (!m) return x;
				if (m[2] == '2') return m[1] + '2';
				if (m[2] == "'") return m[1];
				return m[1] + "'";
			});
		}
		var cc = new mathlib.CubieCube();
		inv("R U R' U R U2 R'".split(/\s+/)).forEach(function(t) { cc.selfMoveStr(t); });
		var ll = cc.toFaceCube();
		var h = cubeCopilot.getHint(ll, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 2});
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [ll]);
		return {stage: cubeCopilot.getStage(ll), kind: h.kind, alg: h.alg};
	});
	console.log('ll', JSON.stringify(info));
	if (info.stage != 'll') fail('test state is not a last-layer state');

	// much shorter than cpTimeout=30
	await page.waitForTimeout(2000);
	var shown = await overlay(page);
	console.log('overlay', JSON.stringify(shown));
	if (!shown.visible) fail('no last-layer hint right after CF');
	if (shown.text.indexOf(info.alg) < 0) fail('wrong last-layer hint: ' + shown.text);

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
