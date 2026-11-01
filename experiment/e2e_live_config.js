"use strict";
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');
var url = process.argv[2] || 'https://alex-beng.github.io/cstimer/feat-cube-copilot/';

(async function() {
	var b = await pw.chromium.launch();
	var p = await b.newPage();
	await p.goto(url, {waitUntil: 'load'});
	await p.waitForTimeout(3500);

	var info = await p.evaluate(function() {
		var chk = document.querySelectorAll('#copilotDiv input[type=checkbox]');
		return {
			inOptions: ('copilot' in MODULE_NAMES),
			label: MODULE_NAMES.copilot,
			checkbox: chk.length,
			def: kernel.getProp('cpEnabled', null),
			timeout: kernel.getProp('cpTimeout', null),
			depth: kernel.getProp('cpF2LDepth', null)
		};
	});
	console.log('live config', JSON.stringify(info));

	var tabs = await p.evaluate(function() {
		var btn = document.querySelectorAll('#leftbar .c1')[0];
		if (btn) btn.click();
		var out = [];
		var list = document.querySelectorAll('.tab');
		for (var i = 0; i < list.length; i++) {
			var t = (list[i].textContent || '').trim();
			if (t.indexOf('Copilot') >= 0) out.push(t);
		}
		return out;
	});
	console.log('live copilot tab', JSON.stringify(tabs));

	var toggle = await p.evaluate(function() {
		var chk = document.querySelectorAll('#copilotDiv input[type=checkbox]')[0];
		if (!chk) return {err: 'no checkbox'};
		chk.click();
		return {enabled: kernel.getProp('cpEnabled', null)};
	});
	console.log('live toggle', JSON.stringify(toggle));

	await b.close();
})();
