"use strict";

// Regression test for virtual-cube solved detection with queued moves.
// If the cube is one U turn from solved and the user turns U twice quickly,
// the first U completes the solve; the second one must not hide it.  The
// timer has to stop on the first U instead of waiting for the whole queue.
//
// Usage: node experiment/e2e_vrc_queued_solve.js
var http = require('http');
var fs = require('fs');
var path = require('path');
var pw = require('C:/Users/Administrator/AppData/Local/Temp/opencode/node_modules/playwright');

var root = path.resolve(__dirname, '..', 'dist', 'local');
var port = 8210;
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
	var page = await browser.newPage({viewport: {width: 1200, height: 900}});
	await page.goto('http://localhost:' + port + '/', {waitUntil: 'load'});
	await page.waitForTimeout(2000);

	await page.evaluate(function() {
		var orig = puzzleFactory.init;
		puzzleFactory.init = function(options, ml, parent, cb) {
			return orig(options, ml, parent, function(ret, isInit) {
				if (ret) window.__puzzle = ret;
				cb(ret, isInit);
			});
		};
		kernel.setProp('vrcSpeed', 0);
		kernel.setProp('vrcMP', 'n');
		kernel.setProp('input', 'v');
	});
	await page.waitForTimeout(2500);

	var scene = await page.evaluate(function() {
		if (!window.__puzzle) {
			return {err: 'no virtual cube'};
		}
		function solve() {
			var sol = min2phase.solve(__puzzle.getFacelet());
			if (sol && !/^Error/.test(sol)) {
				__puzzle.applyMoves(__puzzle.parseScramble(sol, true));
			}
		}
		solve();
		__puzzle.applyMoves(__puzzle.parseScramble('U U U', true)); // U'
		var states = [];
		function listener(move, mstep) {
			if (mstep != 2) {
				return;
			}
			states.push({
				batch: __puzzle.twistyScene.isMoveBatchFinished(),
				full: __puzzle.twistyScene.isMoveFinished(),
				progress: cubeutil.getProgress(__puzzle.getFacelet(), 'n'),
				solved: __puzzle.isSolved('n')
			});
		}
		var scene = __puzzle.twistyScene;
		scene.addMoveListener(listener);
		__puzzle.applyMoves(__puzzle.parseScramble('U U', true));
		return {states: states};
	});
	console.log('scene', JSON.stringify(scene));
	if (scene.err) {
		fail(scene.err);
	} else if (!scene.states.length || scene.states[0].progress != 0 || scene.states[0].solved != 0 ||
		!scene.states[0].batch || scene.states[0].full) {
		fail('the first queued U was not reported as solved');
	}

	if (!scene.err) {
		// drive a real timer run with animated moves: the first queued U must
		// stop the timer even though the second U is already queued
		await page.evaluate(function() {
			kernel.setProp('vrcSpeed', 100);
			timer.virtual.onkeydown(32, {});
		});
		await page.waitForTimeout(500);
		var timerSetup = await page.evaluate(function() {
			var st0 = timer.status();
			timer.status(0);
			var sol = min2phase.solve(__puzzle.getFacelet());
			if (sol && !/^Error/.test(sol)) {
				__puzzle.applyMoves(__puzzle.parseScramble(sol, true));
			}
			__puzzle.applyMoves(__puzzle.parseScramble('U U U', true)); // U'
			timer.startTime($.now());
			timer.curTime([0]);
			timer.status(1);
			__puzzle.addMoves(__puzzle.parseScramble('U U', true));
			return {before: st0};
		});
		await page.waitForTimeout(800);
		var timerRun = await page.evaluate(function() {
			return {
				status: timer.status(),
				facelet: __puzzle.getFacelet()
			};
		});
		timerRun.before = timerSetup.before;
		console.log('timer', JSON.stringify(timerRun));
		if (timerRun.before != -2) {
			fail('virtual timer did not reach the ready state');
		}
		if (timerRun.status != -1) {
			fail('timer did not stop on the first queued U (status ' + timerRun.status + ')');
		}
	}

	await browser.close();
	server.close();
	console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
	process.exitCode = failures == 0 ? 0 : 1;
})().catch(function(e) {
	console.error(e);
	process.exit(1);
});
