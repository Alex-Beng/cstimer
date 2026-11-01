"use strict";

// Smoke test the deployed GitHub Pages build.
// Usage: node experiment/e2e_live.js [url]
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var url = process.argv[2] || 'https://alex-beng.github.io/cstimer/feat-cube-copilot/';
var failures = 0;

function fail(m) {
	failures++;
	console.log('FAIL ' + m);
}

(async function() {
	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	var bad = [];
	page.on('pageerror', function(e) {
		var m = e.message || '';
		if (/ServiceWorker|sw\.js|Wake Lock|bad HTTP response code/i.test(m)) return;
		bad.push('pageerror: ' + m);
	});
	page.on('response', function(r) {
		if (r.status() >= 400 && !/bug\.php|stat\.php|sw\.js/.test(r.url())) {
			bad.push('http ' + r.status() + ': ' + r.url());
		}
	});

	await page.goto(url, {waitUntil: 'load'});
	await page.waitForTimeout(3500);

	var info = await page.evaluate(function() {
		return {
			copilot: typeof cubeCopilot != 'undefined',
			gate: typeof scrambleGate != 'undefined',
			ui: typeof copilotUI != 'undefined',
			div: typeof $ != 'undefined' ? $('#copilotDiv').length : -1,
			scripts: document.querySelectorAll('script[src]').length
		};
	});
	console.log('live info', JSON.stringify(info));

	if (!info.copilot) fail('cubeCopilot not loaded');
	if (!info.gate) fail('scrambleGate not loaded');
	if (!info.ui) fail('copilotUI not loaded');
	if (info.div != 1) fail('copilot window missing');

	if (info.copilot) {
		var r = await page.evaluate(function() {
			var lut = cubeCopilot.buildAlllLut();
			var m = new mathlib.CubieCube();
			['R', 'U', "R'", "U'"].forEach(function(t) { m.selfMoveStr(t); });
			var hint = cubeCopilot.getHint(m.toFaceCube(), {maxDepth: 6, timeLimit: 4000});
			return {lut: lut, hint: hint};
		});
		console.log('lut', JSON.stringify(r.lut));
		console.log('hint', JSON.stringify(r.hint));
		if (!r.hint || r.hint.stage != 'f2l') fail('F2L hint failed on live build');
	}

	bad.slice(0, 8).forEach(fail);

	await browser.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
