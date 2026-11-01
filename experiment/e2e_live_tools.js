"use strict";
// Live check: is Copilot present in the Tools menu on the deployed build?
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');
var url = process.argv[2] || 'https://alex-beng.github.io/cstimer/feat-cube-copilot/';

(async function() {
	var b = await pw.chromium.launch();
	var p = await b.newPage();
	var failures = 0;
	await p.goto(url, {waitUntil: 'load'});
	await p.waitForTimeout(3500);

	await p.evaluate(function() {
		var btn = document.querySelectorAll('#leftbar .c6')[0];
		if (btn) btn.click();
	});
	await p.waitForTimeout(500);

	var res = await p.evaluate(function() {
		var sels = document.querySelectorAll('#toolsDiv select');
		var found = false;
		var opts = [];
		for (var i = 0; i < sels.length; i++) {
			for (var j = 0; j < sels[i].options.length; j++) {
				var t = (sels[i].options[j].textContent || '').trim();
				if (opts.indexOf(t) < 0) opts.push(t);
				if (t.indexOf('Copilot') >= 0) found = true;
			}
		}
		return {found: found, options: opts};
	});
	console.log('live tools has Copilot:', res.found);
	if (!res.found) {
		failures++;
		console.log('available tools:', JSON.stringify(res.options));
	}

	// select it and confirm the panel renders
	var panel = await p.evaluate(function() {
		var sels = document.querySelectorAll('#toolsDiv select');
		for (var i = 0; i < sels.length; i++) {
			for (var j = 0; j < sels[i].options.length; j++) {
				if ((sels[i].options[j].textContent || '').indexOf('Copilot') >= 0) {
					sels[i].selectedIndex = j;
					var ev = document.createEvent('HTMLEvents');
					ev.initEvent('change', true, true);
					sels[i].dispatchEvent(ev);
					return true;
				}
			}
		}
		return false;
	});
	await p.waitForTimeout(500);
	var ui = await p.evaluate(function() {
		var div = document.querySelector('#toolsDiv');
		var chk = div ? div.querySelectorAll('input[type=checkbox]') : [];
		var btn = div ? div.querySelectorAll('input[type=button]') : [];
		var labels = [];
		for (var i = 0; i < btn.length; i++) labels.push(btn[i].value);
		return {checkbox: chk.length, buttons: labels};
	});
	console.log('live panel:', JSON.stringify(ui));
	if (!panel) failures++;
	if (ui.checkbox < 1) failures++;
	if (ui.buttons.length < 1) failures++;

	await b.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
