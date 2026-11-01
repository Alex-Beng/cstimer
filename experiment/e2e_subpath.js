"use strict";

// Verify the src_local build works when served under a sub-path, which is how
// GitHub Pages publishes non-master branches (/feat-cube-copilot/).
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var prefix = '/feat-cube-copilot';
var port = 8151;
var failures = 0;

function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
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
	if (url == prefix) {
		res.writeHead(302, {'Location': prefix + '/'});
		res.end();
		return;
	}
	if (url.indexOf(prefix + '/') != 0) {
		res.writeHead(404);
		res.end('outside prefix: ' + url);
		return;
	}
	var rel = url.slice(prefix.length + 1);
	if (rel == '' || rel == 'index.html') {
		rel = 'index.html';
	}
	var file = path.join(root, rel.replace(/^\/+/, ''));
	if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
		res.writeHead(404);
		res.end('not found');
		return;
	}
	res.writeHead(200, {'Content-Type': MIME[path.extname(file)] || 'application/octet-stream'});
	fs.createReadStream(file).pipe(res);
});

(async function() {
	await new Promise(function(resolve) {
		server.listen(port, '0.0.0.0', function() {
			resolve();
		});
	});

	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	var errors = [];
	page.on('pageerror', function(e) {
		var msg = e.message || '';
		if (/ServiceWorker|sw\.js|Wake Lock|bad HTTP response code/i.test(msg)) return;
		errors.push(msg);
	});
	page.on('response', function(r) {
		if (r.status() >= 400) {
			var u = r.url();
			// bug.php / sw.js are not part of this mode
			if (/bug\.php|sw\.js/.test(u)) return;
			errors.push('http ' + r.status() + ': ' + u);
		}
	});

	await page.goto('http://localhost:' + port + prefix + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var info = await page.evaluate(function() {
		return {
			hasCopilot: typeof cubeCopilot != 'undefined',
			hasGate: typeof scrambleGate != 'undefined',
			hasUI: typeof copilotUI != 'undefined',
			div: $('#copilotDiv').length,
			scripts: document.querySelectorAll('script[src]').length
		};
	});
	console.log('subpath info', JSON.stringify(info));
	if (!info.hasCopilot) fail('cubeCopilot missing under subpath');
	if (!info.hasGate) fail('scrambleGate missing under subpath');
	if (!info.hasUI) fail('copilotUI missing under subpath');

	var hint = await page.evaluate(function() {
		var m = new mathlib.CubieCube();
		['R', 'U', "R'", "U'"].forEach(function(t) { m.selfMoveStr(t); });
		return cubeCopilot.getHint(m.toFaceCube(), {maxDepth: 6, timeLimit: 4000});
	});
	console.log('hint under subpath', JSON.stringify(hint));
	if (!hint || hint.stage != 'f2l') fail('F2L hint broken under subpath');

	var ll = await page.evaluate(function() {
		// a known last-layer case: U-perm
		var m = new mathlib.CubieCube();
		['R', 'U', "R'", "U'", 'R', 'U', "R'", 'U', "R'", 'F', 'R2', "U'", "R'", "U'", 'R', 'U', "R'", "F'"].forEach(function(t) { m.selfMoveStr(t); });
		return cubeCopilot.getStage(m.toFaceCube());
	});
	console.log('stage probe', ll);

	errors.slice(0, 8).forEach(function(e) {
		fail(e);
	});

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
