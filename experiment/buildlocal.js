"use strict";

// Build a static dist/local for local browser testing without PHP/make.
// Mirrors Makefile's src_local target: unminified sources + generated index.

var fs = require('fs');
var path = require('path');

var root = path.resolve(__dirname, '..');
var out = path.join(root, 'dist', 'local');

function copyDir(src, dst) {
	fs.mkdirSync(dst, {recursive: true});
	fs.readdirSync(src, {withFileTypes: true}).forEach(function(e) {
		var s = path.join(src, e.name);
		var d = path.join(dst, e.name);
		if (e.isDirectory()) {
			copyDir(s, d);
		} else {
			fs.copyFileSync(s, d);
		}
	});
}

copyDir(path.join(root, 'src', 'js'), path.join(out, 'js'));
copyDir(path.join(root, 'src', 'css'), path.join(out, 'css'));
copyDir(path.join(root, 'src', 'lang'), path.join(out, 'lang'));
fs.copyFileSync(path.join(root, 'dist', 'js', 'jquery.min.js'), path.join(out, 'js', 'jquery.min.js'));

function readLang(name) {
	return fs.readFileSync(path.join(root, 'src', 'lang', name), 'utf8');
}

// 1. index.php: drop the two PHP includes, inject English strings verbatim.
var index = fs.readFileSync(path.join(root, 'src', 'index.php'), 'utf8');

var titleBlock = [
	'  <meta name="keywords" content="timer, cstimer, rubiks cube timer, online timer, web timer">',
	"  <title> csTimer - Professional Rubik's Cube Speedsolving/Training Timer </title>",
	'  <script type="text/javascript">',
	"var CSTIMER_VERSION = 'local';",
	"var LANG_SET = '|en-us';",
	"var LANG_STR = 'English';",
	"var LANG_CUR = 'en-us';",
	readLang('en-us.js'),
	'  </script>'
].join('\n');

index = index.replace(/<\?php include\('lang\/langDet\.php'\);? ?\?>/, titleBlock);

// the body-level language block (line ~123)
var bodyLang = readLang('en-us.php').replace(/<\?php[\s\S]*?\?>/g, '').replace(/<\?php[\s\S]*$/g, '');
index = index.replace(/<\?php include\('lang\/'\s*\.\s*\$lang\s*\.\s*'\.php'\)\s*\?>/, bodyLang);

// anything left over
index = index.replace(/<\?php[\s\S]*?\?>/g, '');
index = index.replace(/<\?php[\s\S]*$/g, '');

// 2. register the new modules in the unminified load order
index = index.replace(
	'<script type="text/javascript" src="js/lib/puzzlefactory.js"></script>',
	'<script type="text/javascript" src="js/lib/puzzlefactory.js"></script>\n' +
	'  <script type="text/javascript" src="js/lib/scrambleready.js"></script>\n' +
	'  <script type="text/javascript" src="js/lib/alll.js"></script>\n' +
	'  <script type="text/javascript" src="js/lib/copilot.js"></script>'
);
index = index.replace(
	'<script type="text/javascript" src="js/tools/metronome.js"></script>',
	'<script type="text/javascript" src="js/tools/metronome.js"></script>\n' +
	'  <script type="text/javascript" src="js/tools/copilotui.js"></script>'
);

fs.writeFileSync(path.join(out, 'index.html'), index);

var checks = {
	scrdata: index.indexOf('var scrdata') != -1,
	langCur: index.indexOf('LANG_CUR') != -1,
	copilot: index.indexOf('js/lib/copilot.js') != -1,
	copilotui: index.indexOf('js/tools/copilotui.js') != -1,
	phpLeft: (index.match(/<\?php/g) || []).length
};
console.log('local build ready at', out);
console.log(JSON.stringify(checks));
