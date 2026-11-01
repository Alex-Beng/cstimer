"use strict";
// Live check: an idle F2L position must show a VISIBLE hint during a solve,
// where csTimer fades #leftbar (logo banner) to 0.01 opacity.
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');
var url = process.argv[2] || 'https://alex-beng.github.io/cstimer/feat-cube-copilot/';
var failures = 0;

(async function() {
	var b = await pw.chromium.launch();
	var p = await b.newPage();
	await p.goto(url, {waitUntil: 'load'});
	await p.waitForTimeout(3500);

	await p.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 3);
		kernel.setProp('cpF2LDepth', 6);
		var m = new mathlib.CubieCube();
		['R', 'U', "R'"].forEach(function(t) { m.selfMoveStr(t); });
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [m.toFaceCube()]);
		kernel.ui.setAutoShow(false);
	});

	await p.waitForTimeout(12000);

	var res = await p.evaluate(function() {
		var o = document.querySelector('#copilotHint');
		var lb = document.querySelector('#leftbar');
		var cs = o ? getComputedStyle(o) : null;
		return {
			display: cs ? cs.display : null,
			text: (o ? o.textContent : '').replace(/\s+/g, ' ').trim(),
			leftbarOpacity: lb ? getComputedStyle(lb).opacity : null
		};
	});
	console.log('live idle hint', JSON.stringify(res));

	if (!res.display || res.display === 'none') {
		failures++;
		console.log('FAIL: hint overlay not visible on the live build');
	}
	if (!/F2L/.test(res.text) || !/[URFDLB]/.test(res.text)) {
		failures++;
		console.log('FAIL: overlay has no algorithm');
	}

	await b.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
