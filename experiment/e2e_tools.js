"use strict";

// Verify the Copilot is reachable from the Tools menu and that its panel works.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8171;
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

(async function() {
	await new Promise(function(r) {
		server.listen(port, '0.0.0.0', r);
	});

	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	// open the Tools window/panel
	var opened = await page.evaluate(function() {
		// the tools module exposes its window via the left bar button (index 6)
		var btn = document.querySelectorAll('#leftbar .c6')[0];
		if (btn) btn.click();
		return !!btn;
	});
	console.log('tools button clicked:', opened);
	await page.waitForTimeout(600);

	// check the tool menu contains a Copilot entry
	var menu = await page.evaluate(function() {
		var out = {selects: [], selected: null, hasCopilot: false, panelText: ''};
		var sels = document.querySelectorAll('#toolsDiv select');
		for (var i = 0; i < sels.length; i++) {
			var opts = [];
			for (var j = 0; j < sels[i].options.length; j++) {
				opts.push(sels[i].options[j].textContent.trim());
			}
			out.selects.push(opts);
			if (opts.join('|').indexOf('Copilot') >= 0) out.hasCopilot = true;
		}
		out.panelText = ($('#toolsDiv').text() || '').slice(0, 300);
		return out;
	});
	console.log('tool selects:', JSON.stringify(menu.selects));
	console.log('tools panel text:', menu.panelText.replace(/\s+/g, ' ').slice(0, 200));
	if (!menu.hasCopilot) fail('Copilot missing from Tools menu');

	// select the copilot tool and verify its panel renders controls
	var panel = await page.evaluate(function() {
		var sels = document.querySelectorAll('#toolsDiv select');
		for (var i = 0; i < sels.length; i++) {
			for (var j = 0; j < sels[i].options.length; j++) {
				if ((sels[i].options[j].textContent || '').indexOf('Copilot') >= 0) {
					sels[i].selectedIndex = j;
					sels[i].onchange && sels[i].onchange({target: sels[i]});
					var ev = document.createEvent('HTMLEvents');
					ev.initEvent('change', true, true);
					sels[i].dispatchEvent(ev);
					return {ok: true, idx: i};
				}
			}
		}
		return {ok: false};
	});
	console.log('select copilot:', JSON.stringify(panel));

	await page.waitForTimeout(600);
	var ui = await page.evaluate(function() {
		var div = document.querySelector('#toolsDiv');
		var checks = div ? div.querySelectorAll('input[type=checkbox]') : [];
		var buttons = div ? div.querySelectorAll('input[type=button]') : [];
		var labels = [];
		for (var i = 0; i < buttons.length; i++) labels.push(buttons[i].value);
		return {
			checkboxes: checks.length,
			buttons: labels,
			text: (div ? div.textContent : '').replace(/\s+/g, ' ').slice(0, 240)
		};
	});
	console.log('tool panel ui:', JSON.stringify(ui));
	if (ui.checkboxes < 1) fail('copilot tool panel has no enable checkbox');
	if (ui.buttons.length < 1) fail('copilot tool panel has no hint button');

	// toggling the tool checkbox must enable the feature
	var toggled = await page.evaluate(function() {
		var div = document.querySelector('#toolsDiv');
		var chk = div.querySelectorAll('input[type=checkbox]')[0];
		if (!chk) return {err: 'no checkbox'};
		chk.click();
		return {enabled: kernel.getProp('cpEnabled', null)};
	});
	console.log('tool toggle:', JSON.stringify(toggled));
	if (toggled.enabled !== true) fail('tool checkbox did not enable cpEnabled');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
