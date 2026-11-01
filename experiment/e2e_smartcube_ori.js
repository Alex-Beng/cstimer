"use strict";

// Smart-cube hint frame and multi-pair F2L hints.
//
//  - once the cross is solved the hint must be written with that cross at the
//    bottom and a fixed front colour (the "Cube orientation" setting is not
//    used any more)
//  - the same frame must be found whatever whole-cube orientation the cube
//    reports
//  - several unsolved F2L pairs can be hinted at once
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8197;
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

var MULTI = 'FUULUUULRUFFURUBRFLFBBFBFFRDDDDDDDDDLBBRLRLLLRRULBFRBB';

(async function() {
	await new Promise(function(r) {
		server.listen(port, '0.0.0.0', r);
	});
	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var setup = await page.evaluate(function(multi) {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 0);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpCustom', '');
		kernel.setProp('cpF2LCount', 1);
		cubeCopilot.warmUp();

		var cc = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { cc.selfMoveStr(t); });
		var raw = cc.toFaceCube();             // cross solved, white on top

		var sa = cubeCopilot.pickSmartOrientation(raw);
		var ha = cubeCopilot.getHint(sa.facelet, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 1});
		var hm = cubeCopilot.getHint(cubeCopilot.pickSmartOrientation(multi).facelet,
			{maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 4});

		window.__raw = raw;
		window.__multiAlg = hm.pairs ? hm.pairs[0].alg : '';
		var C = {U: 'White', R: 'Red', F: 'Green', D: 'Yellow', L: 'Orange', B: 'Blue'};
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('giikerState', [raw]);
		return {
			a: {d: sa.d, f: sa.f, alg: ha.alg},
			oriText: C[sa.d] + ' bottom, ' + C[sa.f] + ' front',
			multiPairs: hm.pairs ? hm.pairs.length : 0,
			multiNames: hm.pairs ? hm.pairs.map(function(p) { return p.name; }) : []
		};
	}, MULTI);
	console.log('setup', JSON.stringify(setup));

	if (setup.a.f != 'F') {
		fail('front colour should follow the fixed order (green): ' + JSON.stringify(setup.a));
	}
	if (setup.multiPairs < 2) {
		fail('multi-pair hint returned only ' + setup.multiPairs + ' pair(s)');
	}

	await page.waitForTimeout(1500);
	var shown = await overlay(page);
	console.log('overlay', JSON.stringify(shown));
	if (!shown.visible) fail('no hint for a smart-cube state');
	if (shown.text.indexOf(setup.a.alg) < 0) fail('hint algorithm is not in the stage frame');
	if (shown.text.indexOf(setup.oriText) < 0) fail('hint does not display the frame: ' + shown.text);

	// several F2L pairs at once
	await page.evaluate(function(multi) {
		kernel.setProp('cpF2LCount', 4);
		kernel.pushSignal('giikerState', [multi]);
	}, MULTI);
	await page.waitForTimeout(1500);
	var multiShown = await overlay(page);
	console.log('multi overlay', JSON.stringify(multiShown));
	if (!multiShown.visible) fail('no multi-pair hint');
	if (multiShown.text.indexOf('Next F2L pairs') < 0) fail('multi-pair header missing');
	var named = setup.multiNames.filter(function(n) {
		return multiShown.text.indexOf(n + ':') >= 0;
	});
	if (named.length < 2) fail('multi-pair overlay does not list pairs: ' + multiShown.text);

	// following one of the pairs must not drop the hint
	var follow = await page.evaluate(function(multi) {
		var ori = cubeCopilot.pickSmartOrientation(multi);
		var h = cubeCopilot.getHint(ori.facelet, {maxDepth: 8, timeLimit: 4000, llMode: 'op', f2lCount: 4});
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(ori.facelet);
		h.pairs[0].moves.forEach(function(t) { cc.selfMoveStr(t); });
		kernel.pushSignal('giikerState', [cc.toFaceCube()]);
		return h.pairs[0].alg;
	}, MULTI);
	await page.waitForTimeout(300);
	var afterFollow = await overlay(page);
	console.log('after following', follow, JSON.stringify(afterFollow));
	if (!afterFollow.visible) fail('hint disappeared when following one of the pairs');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
