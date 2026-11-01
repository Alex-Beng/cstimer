"use strict";

// Verify the Copilot is configurable in the UI:
//  - MODULE_NAMES has a "copilot" entry, so the Options panel shows the group
//  - the copilot window has an enable checkbox bound to cpEnabled
//  - default is off, and toggling the checkbox turns it on
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8161;
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

	var info = await page.evaluate(function() {
		var checkbox = $('#copilotDiv input[type=checkbox]');
		return {
			moduleName: typeof MODULE_NAMES != 'undefined' ? MODULE_NAMES.copilot : null,
			inOptions: typeof MODULE_NAMES != 'undefined' && 'copilot' in MODULE_NAMES,
			windowCheckboxes: checkbox.length,
			defaultEnabled: kernel.getProp('cpEnabled', null),
			checkedNow: checkbox.length ? checkbox[0].checked : null,
			hasTimeoutProp: kernel.getProp('cpTimeout', null),
			hasDepthProp: kernel.getProp('cpF2LDepth', null),
			llMode: kernel.getProp('cpLlmode', null),
			f2lCount: kernel.getProp('cpF2LCount', null)
		};
	});
	console.log('config info', JSON.stringify(info));

	if (!info.inOptions) fail('MODULE_NAMES.copilot missing -> options tab hidden');
	if (info.moduleName !== 'Copilot') fail('MODULE_NAMES.copilot label wrong: ' + info.moduleName);
	if (info.defaultEnabled !== false) fail('cpEnabled should default to false, got ' + info.defaultEnabled);
	if (info.checkedNow !== false) fail('window checkbox should start unchecked');
	if (info.windowCheckboxes < 1) fail('copilot window has no enable checkbox');
	if (info.hasTimeoutProp !== 6) fail('cpTimeout default should be 6, got ' + info.hasTimeoutProp);
	if (info.hasDepthProp !== 8) fail('cpF2LDepth default should be 8, got ' + info.hasDepthProp);
	if (info.llMode !== 'op') fail('cpLlmode should default to op, got ' + info.llMode);
	if (info.f2lCount !== 2) fail('cpF2LCount default should be 2, got ' + info.f2lCount);

	// toggling the checkbox must flip the property and the status text
	var after = await page.evaluate(function() {
		$('#copilotDiv input[type=checkbox]').click();
		return {
			enabled: kernel.getProp('cpEnabled', null),
			statusText: $('#copilotDiv').text()
		};
	});
	console.log('after toggle', JSON.stringify(after));
	if (after.enabled !== true) fail('checkbox did not enable cpEnabled');
	if (!/Waiting|等待/.test(after.statusText)) fail('status did not switch to waiting: ' + after.statusText);

	// the options dialog builds its tabs from MODULE_NAMES + registered props
	var opts = await page.evaluate(function() {
		$('#leftbar .c1').click();
		var tabs = [];
		$('.tab').each(function() {
			tabs.push($(this).text().trim());
		});
		var names = [];
		$('.opttable input, .opttable select').each(function() {
			if (this.name) names.push(this.name);
		});
		return {tabs: tabs, inputs: names};
	});
	console.log('options tabs', JSON.stringify(opts.tabs));
	var hasCopilotTab = opts.tabs.some(function(t) {
		return t.indexOf('Copilot') != -1;
	});
	if (!hasCopilotTab) fail('Options panel has no Copilot tab');
	if (opts.inputs.indexOf('cpEnabled') == -1) fail('cpEnabled not rendered in Options');
	if (opts.inputs.indexOf('cpTimeout') == -1) fail('cpTimeout not rendered in Options');
	if (opts.inputs.indexOf('cpF2LDepth') == -1) fail('cpF2LDepth not rendered in Options');
	if (opts.inputs.indexOf('cpLlmode') == -1) fail('cpLlmode not rendered in Options');
	if (opts.inputs.indexOf('cpF2LCount') == -1) fail('cpF2LCount not rendered in Options');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
