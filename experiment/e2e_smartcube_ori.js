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
		window.__saOri = sa.ori;
		window.__vrcCalls = [];
		timer.giiker.setVrcOri = function(r) {
			window.__vrcCalls.push(r);
			return true;
		};
		var C = {U: 'White', R: 'Red', F: 'Green', D: 'Yellow', L: 'Orange', B: 'Blue'};
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('giikerState', [raw]);
		return {
			a: {d: sa.d, f: sa.f, ori: sa.ori, alg: ha.alg},
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

	// after cross + one pair the frame is locked: an algorithm that temporarily
	// breaks the cross must not rotate the hint or the VRC view again
	var broken = await page.evaluate(function() {
		window.__vrcCalls = [];
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(window.__raw);
		cc.selfMoveStr('R'); // breaks the cross
		var f = cc.toFaceCube();
		var det = cubeCopilot.pickSmartOrientation(f);
		kernel.pushSignal('giikerState', [f]);
		return {stage: cubeCopilot.getStage(f), detected: det.ori, cross: det.cross, calls: (window.__vrcCalls || []).slice()};
	});
	console.log('broken-cross', JSON.stringify(broken));
	if (broken.cross) fail('test state did not break the cross (invalid)');
	if (!broken.calls.length || broken.calls[0] != setup.a.ori) {
		fail('locked frame was not kept after the cross was broken: ' + JSON.stringify(broken));
	}
	var still = await overlay(page);
	if (!still.visible) fail('hint disappeared after one breaking move');

	// several F2L pairs at once (a new solve, so the lock is cleared)
	await page.evaluate(function(multi) {
		kernel.pushSignal('timerStatus', -1);
		kernel.pushSignal('timerStatus', 1);
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

	// the smart-cube VRC view must be rotated to the hint frame once the cross
	// is solved, and not before
	var vrc = await page.evaluate(function() {
		return {calls: window.__vrcCalls || [], expect: window.__saOri};
	});
	console.log('vrc calls', JSON.stringify(vrc));
	if (!vrc.calls.length) {
		fail('VRC orientation was not updated after the cross');
	} else if (vrc.calls[0] != vrc.expect) {
		fail('VRC orientation does not match the hint frame');
	}

	var noCross = await page.evaluate(function() {
		window.__vrcCalls = [];
		kernel.pushSignal('scramble', ['333', "R U F D L' B2", 0]);
		var cc = new mathlib.CubieCube();
		["R", 'U', 'F', 'D', "L'", 'B2'].forEach(function(t) { cc.selfMoveStr(t); });
		var f = cc.toFaceCube();
		kernel.pushSignal('giikerState', [f]);
		return {stage: cubeCopilot.getStage(f), calls: window.__vrcCalls.slice()};
	});
	console.log('no-cross', JSON.stringify(noCross));
	if (noCross.stage == 'cross' && noCross.calls.length) {
		fail('VRC orientation changed before the cross was solved');
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
