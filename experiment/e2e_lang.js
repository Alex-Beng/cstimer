"use strict";

// Static-hosting language switching (GitHub Pages has no PHP runtime).
//
//  1. ?lang=zh-cn replaces the baked-in English strings before boot
//  2. the language is persisted in a cookie and reused without the query
//  3. ?lang=cn is an alias for zh-cn
//  4. navigator.language picks a locale when nothing else is set

var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
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
	'.webmanifest': 'application/manifest+json'
};

var server = http.createServer(function(req, res) {
	var url = decodeURIComponent(req.url.split('?')[0]);
	if (url == '/') {
		url = '/index.html';
	}
	var file = path.join(root, url.replace(/^\/+/, ''));
	if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
		res.writeHead(404);
		res.end('not found');
		return;
	}
	res.writeHead(200, {'Content-Type': MIME[path.extname(file)] || 'application/octet-stream'});
	fs.createReadStream(file).pipe(res);
});

async function snapshot(page) {
	return await page.evaluate(function() {
		return {
			lang: window.LANG_CUR,
			htmlLang: document.documentElement.lang,
			about: window.ABOUT_LANG,
			prop: kernel.getProp('lang'),
			cookie: /(?:^|;\s*)lang=([^;]+)/.exec(document.cookie || '')
		};
	});
}

(async function() {
	await new Promise(function(resolve) {
		server.listen(port, '0.0.0.0', resolve);
	});

	var browser = await pw.chromium.launch();
	var errors = [];

	var context = await browser.newContext();
	var page = await context.newPage();
	page.on('pageerror', function(e) {
		var msg = e.message || '';
		if (!/ServiceWorker|sw\.js|Wake Lock|bad HTTP response code/i.test(msg)) {
			errors.push(msg);
		}
	});

	await page.goto('http://localhost:' + port + '/?lang=zh-cn', {waitUntil: 'load'});
	await page.waitForTimeout(600);
	var fromQuery = await snapshot(page);

	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(600);
	var fromCookie = await snapshot(page);

	await page.goto('http://localhost:' + port + '/?lang=cn', {waitUntil: 'load'});
	await page.waitForTimeout(600);
	var alias = await snapshot(page);

	await page.goto('http://localhost:' + port + '/?lang=fr-fr', {waitUntil: 'load'});
	await page.waitForTimeout(600);
	var french = await snapshot(page);

	// Changing the setting in-app navigates to ?lang=..., which the loader
	// must honor on the next static page load.
	var navigated = page.waitForNavigation({waitUntil: 'load'});
	await page.evaluate(function() {
		kernel.setProp('lang', 'zh-cn');
	});
	await navigated;
	await page.waitForTimeout(600);
	var switched = await snapshot(page);

	var autoContext = await browser.newContext({locale: 'ja-JP'});
	var autoPage = await autoContext.newPage();
	await autoPage.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await autoPage.waitForTimeout(600);
	var auto = await snapshot(autoPage);

	await autoContext.close();
	await context.close();
	await browser.close();
	server.close();

	console.log(JSON.stringify({
		query: fromQuery,
		cookie: fromCookie,
		alias: alias,
		french: french,
		switched: switched,
		auto: auto,
		errors: errors
	}, null, 2));

	function checkLang(name, got, expectLang, expectAbout) {
		if (got.lang != expectLang || got.prop != expectLang ||
				got.htmlLang != expectLang || got.about != expectAbout) {
			fail(name + ': ' + JSON.stringify(got));
		}
	}

	checkLang('?lang=zh-cn', fromQuery, 'zh-cn', '关于');
	if (!fromQuery.cookie || fromQuery.cookie[1] != 'zh-cn') {
		fail('?lang=zh-cn did not persist the cookie: ' + JSON.stringify(fromQuery));
	}
	checkLang('cookie fallback', fromCookie, 'zh-cn', '关于');
	checkLang('?lang=cn alias', alias, 'zh-cn', '关于');
	checkLang('?lang=fr-fr', french, 'fr-fr', 'À propos');
	checkLang('in-app switch', switched, 'zh-cn', '关于');
	checkLang('navigator.language', auto, 'ja-jp', '概要');

	errors.forEach(function(msg) {
		fail('page error: ' + msg);
	});

	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
