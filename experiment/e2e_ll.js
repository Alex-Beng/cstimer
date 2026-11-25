"use strict";

// Two-look last layer through the UI: OLL hint, then (in the next position) a
// PLL hint, plus the ALLL fallback, ZBLL / COLL hints and the "off" setting.
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8194;
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
		return {
			visible: !!(cs && cs.display != 'none'),
			text: (o ? o.textContent : '').replace(/\s+/g, ' ').trim()
		};
	});
}

(async function() {
	await new Promise(function(r) {
		server.listen(port, '0.0.0.0', r);
	});
	var browser = await pw.chromium.launch();
	var page = await browser.newPage();
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2500);

	var info = await page.evaluate(function() {
		kernel.setProp('cpEnabled', true);
		kernel.setProp('cpTimeout', 1);
		kernel.setProp('cpLlmode', 'op');
		kernel.setProp('cpDrift', 1);
		kernel.setProp('cpCustom', '');
		cubeCopilot.warmUp();

		function inv(tokens) {
			return tokens.slice().reverse().map(function(t) {
				var m = /^([URFDLB]w?|[EMSxyz])(2|')?$/.exec(t);
				if (!m) return t;
				if (m[2] == '2') return m[1] + '2';
				if (m[2] == "'") return m[1];
				return m[1] + "'";
			});
		}
		function stateFor(alg) {
			var cc = new mathlib.CubieCube();
			inv(alg.split(/\s+/).filter(Boolean)).forEach(function(t) { cc.selfMoveStr(t); });
			return cc.toFaceCube();
		}
		// a last-layer state that this OLL turns into the T-perm position
		function ollThenPll(ollAlg, pllAlg) {
			var cc = new mathlib.CubieCube();
			inv(pllAlg.split(/\s+/).filter(Boolean)).forEach(function(t) { cc.selfMoveStr(t); });
			inv(ollAlg.split(/\s+/).filter(Boolean)).forEach(function(t) { cc.selfMoveStr(t); });
			return cc.toFaceCube();
		}

		var ollState = ollThenPll('R U R\' U R U2 R\'', 'R U R\' U\' R\' F R2 U\' R\' U\' R U R\' F\'');
		var pllState = stateFor('R U R\' U\' R\' F R2 U\' R\' U\' R U R\' F\'');
		window.__ollState = ollState;
		window.__pllState = pllState;

		var ollHint = cubeCopilot.getHint(ollState, {maxDepth: 8, timeLimit: 4000, llMode: 'op'});
		var pllHint = cubeCopilot.getHint(pllState, {maxDepth: 8, timeLimit: 4000, llMode: 'op'});
		var alllHint = cubeCopilot.getHint(ollState, {maxDepth: 8, timeLimit: 4000, llMode: 'alll'});
		window.__infoBase = ollHint.base.join(' ');
		window.__ollFull = (ollHint.pre ? ollHint.pre.split(/\s+/) : []).concat(ollHint.base);
		kernel.pushSignal('timerStatus', 1);
		kernel.pushSignal('virtualState', [ollState]);
		return {
			ollStage: cubeCopilot.getStage(ollState),
			ollKind: ollHint.kind, ollName: ollHint.name, ollKey: ollHint.key,
			ollPre: ollHint.pre, ollBase: ollHint.base,
			pllKind: pllHint.kind, pllName: pllHint.name, pllKey: pllHint.key,
			alllKind: alllHint.kind
		};
	});
	console.log('hints', JSON.stringify(info));
	if (info.ollStage != 'll') fail('OLL test state is not a last-layer state');
	if (info.ollKind != 'oll') fail('expected an OLL hint, got ' + info.ollKind);
	if (info.pllKind != 'pll') fail('expected a PLL hint, got ' + info.pllKind);
	if (info.alllKind != 'alll') fail('expected an ALLL hint in alll mode, got ' + info.alllKind);

	await page.waitForTimeout(6000);
	var ollShown = await overlay(page);
	console.log('oll overlay', JSON.stringify(ollShown));
	if (!ollShown.visible) fail('OLL hint not shown');
	if (ollShown.text.indexOf(info.ollName) < 0) fail('overlay does not name the OLL case');

	// save a custom alg for this OLL case equal to the engine base
	var save = await page.evaluate(function() {
		var div = document.querySelector('#copilotDiv');
		var input = div.querySelector('input[type=text]');
		var buttons = [].slice.call(div.querySelectorAll('input[type=button]'));
		var saveBtn = buttons.filter(function(b) { return /save/i.test(b.value); })[0];
		input.value = window.__infoBase;
		saveBtn.click();
		return kernel.getProp('cpCustom');
	});
	console.log('custom map', save);

	// a valid alternative algorithm must replace the displayed one
	var alt = await page.evaluate(function() {
		var text = "R U R' U R U2 R' D U D' U'";
		var div = document.querySelector('#copilotDiv');
		var input = div.querySelector('input[type=text]');
		var buttons = [].slice.call(div.querySelectorAll('input[type=button]'));
		var saveBtn = buttons.filter(function(b) { return /save/i.test(b.value); })[0];
		input.value = text;
		saveBtn.click();
		window.__ollFull = text.split(/\s+/);
		return kernel.getProp('cpCustom');
	});
	console.log('custom map (alt)', alt);
	var altShown = await overlay(page);
	console.log('alt overlay', JSON.stringify(altShown));
	if (!altShown.visible) fail('hint disappeared after saving an alternative algorithm');
	if (altShown.text.indexOf("R U R' U R U2 R' D U D' U'") < 0) {
		fail('the custom algorithm is not displayed: ' + altShown.text);
	}

	// solve the OLL case, which must complete (and hide) the hint...
	var ollSteps = await page.evaluate(function() {
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(window.__ollState);
		var ret = [];
		window.__ollFull.slice().forEach(function(t) {
			cc.selfMoveStr(t);
			var facelet = cc.toFaceCube();
			ret.push({facelet: facelet, stage: cubeCopilot.getStage(facelet)});
		});
		return ret;
	});
	// the algorithm passes through cross/F2L states; the OLL hint must stay
	for (var si = 0; si < ollSteps.length - 1; si++) {
		await page.evaluate(function(st) {
			kernel.pushSignal('virtualState', [st.facelet]);
		}, ollSteps[si]);
		await page.waitForTimeout(150);
		var mid = await overlay(page);
		if (!mid.visible) {
			fail('hint disappeared after OLL step ' + (si + 1) + ' (' + ollSteps[si].stage + ')');
		}
	}
	var afterOll = ollSteps.length ? ollSteps[ollSteps.length - 1].stage : null;
	await page.evaluate(function(st) {
		kernel.pushSignal('virtualState', [st.facelet]);
	}, ollSteps[ollSteps.length - 1]);
	console.log('after OLL', afterOll);
	if (afterOll != 'll') fail('OLL algorithm did not leave a last-layer state');

	// ...and the position it reaches must get a PLL hint
	await page.waitForTimeout(6000);
	var pllShown = await overlay(page);
	console.log('pll overlay', JSON.stringify(pllShown));
	if (!pllShown.visible) fail('PLL hint not shown after OLL');
	if (!/PLL/.test(pllShown.text)) fail('overlay does not show a PLL hint: ' + pllShown.text);

	// ZBLL / COLL hints use their own algorithm sets
	var special = await page.evaluate(function() {
		function inverseFacelet(alg) {
			var cc = new mathlib.CubieCube();
			alg.split(/\s+/).reverse().forEach(function(t) {
				var m = /^([URFDLB]w?|[EMS])(2|')?$/.exec(t);
				if (!m) return;
				var mod = m[2] || '';
				var inv = mod == '' ? "'" : (mod == "'" ? '' : '2');
				cc.selfMoveStr(m[1] + inv);
			});
			return cc.toFaceCube();
		}
		function firstState(list, mode) {
			for (var i = 0; i < list.length; i++) {
				var alg = list[i].split('|')[1];
				if (!alg || /[xyz]/.test(alg)) continue;
				var f = inverseFacelet(alg);
				var h = cubeCopilot.getHint(f, {timeLimit: 4000, llMode: mode});
				if (h && h.kind == mode) {
					return {facelet: f, kind: h.kind, name: h.name};
				}
			}
			return null;
		}
		return {
			zbll: firstState(ZBLL_DATA, 'zbll'),
			coll: firstState(COLL_DATA, 'coll')
		};
	});
	console.log('special', JSON.stringify(special));
	if (!special.zbll || special.zbll.kind != 'zbll') fail('no ZBLL hint found');
	if (!special.coll || special.coll.kind != 'coll') fail('no COLL hint found');

	if (special.zbll) {
		await page.evaluate(function(st) {
			kernel.setProp('cpLlmode', 'zbll');
			kernel.pushSignal('virtualState', [st.facelet]);
		}, special.zbll);
		await page.waitForTimeout(6000);
		var zb = await overlay(page);
		console.log('zbll overlay', JSON.stringify(zb));
		if (!zb.visible || !/ZBLL/.test(zb.text)) fail('ZBLL hint not shown: ' + zb.text);
	}
	if (special.coll) {
		await page.evaluate(function(st) {
			kernel.setProp('cpLlmode', 'coll');
			kernel.pushSignal('virtualState', [st.facelet]);
		}, special.coll);
		await page.waitForTimeout(6000);
		var co = await overlay(page);
		console.log('coll overlay', JSON.stringify(co));
		if (!co.visible || !/COLL/.test(co.text)) fail('COLL hint not shown: ' + co.text);
	}

	// "off" suppresses the last-layer hint
	await page.evaluate(function() {
		kernel.setProp('cpLlmode', 'off');
		kernel.pushSignal('virtualState', [window.__ollState]);
	});
	await page.waitForTimeout(6000);
	var offShown = await overlay(page);
	console.log('off overlay', JSON.stringify(offShown));
	if (offShown.visible) fail('last-layer hint shown although the mode is off');

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})();
