"use strict";

// Build the main csTimer bundle with Closure Compiler, mirroring the Makefile
// source list.  Useful on Windows where `make` is unavailable.
// Usage: node experiment/build.js [--check] [--out path]

var fs = require('fs');
var path = require('path');
var child = require('child_process');

var root = path.resolve(__dirname, '..');
var mk = fs.readFileSync(path.join(root, 'Makefile'), 'utf8');

function srcList(name) {
	var start = mk.indexOf(name + ' = ');
	var end = mk.indexOf('\n\n', start);
	var block = mk.slice(start, end < 0 ? mk.length : end);
	// entries end with "\" except the last one, which is followed by ")"
	return [...block.matchAll(/^\s*([\w\/.]+\.js)\s*\\?\s*\)?\s*$/gm)].map(function(m) {
		return path.join('src', 'js', m[1]).replace(/\\/g, '/');
	});
}

var args = process.argv.slice(2);
var isCheck = args.indexOf('--check') != -1;
var outIdx = args.indexOf('--out');
var out = outIdx != -1 ? args[outIdx + 1] : 'dist/js/cstimer.js';

var java = process.env.JAVA_HOME ? path.join(process.env.JAVA_HOME, 'bin', 'java.exe') : 'java';
var compiler = path.join(root, 'lib', 'compiler.jar');

var timerSrc = srcList('timerSrc');
var twistySrc = srcList('twistySrc');

function run(files, output, extra) {
	var argv = [
		'-jar', compiler,
		'--use_types_for_optimization',
		'--language_out', 'STABLE',
		'--charset', 'UTF-8',
		'--strict_mode_input'
	].concat(extra || []);
	if (isCheck) {
		argv = argv.concat([
			'--externs', path.join(root, 'experiment', 'checkwrap.js'),
			path.join(root, 'src', 'lang', 'en-us.js')
		], files, ['-O', 'ADVANCED', '--checks-only', '--jscomp_off', 'checkTypes']);
	} else {
		argv = argv.concat([
			"--define=DEBUGM=false",
			"--define=DEBUGWK=false",
			"--define=ISCSTIMER=true",
			'--js_output_file', output
		], files);
	}
	var res = child.spawnSync(java, argv, {cwd: root, encoding: 'utf8'});
	var text = (res.stdout || '') + (res.stderr || '');
	if (text.trim()) {
		console.log(text.trim());
	}
	return res.status;
}

var status = run(timerSrc, path.join(root, out));
console.log('timer bundle:', status == 0 ? 'OK' : 'FAILED');
if (status != 0) {
	process.exitCode = status;
	return;
}
if (!isCheck) {
	var twistyStatus = run(twistySrc, path.join(root, 'dist', 'js', 'twisty.js'));
	console.log('twisty bundle:', twistyStatus == 0 ? 'OK' : 'FAILED');
	process.exitCode = twistyStatus;
}
