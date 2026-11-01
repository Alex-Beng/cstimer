"use strict";

// Faithful Node emulation of `make src_local` for local verification when PHP
// and/or make are unavailable.  Produces dist/local exactly like the CI does.
var fs = require('fs');
var path = require('path');

var root = path.resolve(__dirname, '..');
var out = path.join(root, 'dist', 'local');

function rmrf(p) {
	fs.rmSync(p, {recursive: true, force: true});
}

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

rmrf(out);
fs.mkdirSync(path.join(out, 'js'), {recursive: true});
fs.mkdirSync(path.join(out, 'css'), {recursive: true});
fs.mkdirSync(path.join(out, 'lang'), {recursive: true});

copyDir(path.join(root, 'src', 'js'), path.join(out, 'js'));
fs.copyFileSync(path.join(root, 'src', 'css', 'style.css'), path.join(out, 'css', 'style.css'));

fs.readdirSync(path.join(root, 'src', 'lang')).forEach(function(f) {
	if (/\.(js|php)$/.test(f)) {
		fs.copyFileSync(path.join(root, 'src', 'lang', f), path.join(out, 'lang', f));
	}
});

fs.copyFileSync(path.join(root, 'dist', 'cstimer.webmanifest'), path.join(out, 'cstimer.webmanifest'));
fs.copyFileSync(path.join(root, 'dist', 'cstimer512x512.png'), path.join(out, 'cstimer512x512.png'));
fs.copyFileSync(path.join(root, 'dist', 'js', 'jquery.min.js'), path.join(out, 'js', 'lib', 'jquery-1.8.0.js'));

// php index.php  -> index.html  (only the tags and the inline language script
// matter for the app to boot)
var enUsJs = fs.readFileSync(path.join(root, 'src', 'lang', 'en-us.js'), 'utf8');
var enUsPhp = fs.readFileSync(path.join(root, 'src', 'lang', 'en-us.php'), 'utf8');

// Render a PHP template the way the server would for lang=en-us: substitute the
// includes/echoes first, then drop the remaining control-flow blocks.
function renderLangTemplate(tpl) {
	tpl = tpl.replace(/<\?php include\('lang\/'\s*\.\s*\$lang\s*\.\s*'\.js'\);?\s*\?>/g, enUsJs);
	tpl = tpl.replace(/<\?php echo \$version; \?>/g, 'Unspecified');
	tpl = tpl.replace(/<\?php echo \$lang; \?>/g, 'en-us');
	tpl = tpl.replace(/<\?php[\s\S]*?\?>/g, '');
	return tpl;
}

var langDet = renderLangTemplate(fs.readFileSync(path.join(root, 'src', 'lang', 'langDet.php'), 'utf8'));

var php = fs.readFileSync(path.join(root, 'src', 'index.php'), 'utf8');
php = php.replace(/<\?php include\('lang\/langDet\.php'\);? ?\?>/, langDet);
php = php.replace(/<\?php include\('lang\/'\s*\.\s*\$lang\s*\.\s*'\.js'\);?\s*\?>/, enUsJs);
php = php.replace(/<\?php include\('lang\/'\s*\.\s*\$lang\s*\.\s*'\.php'\)\s*\?>/,
	renderLangTemplate(enUsPhp));
php = php.replace(/<\?php echo \$version; \?>/g, 'Unspecified');
php = php.replace(/<\?php echo \$lang; \?>/g, 'en-us');
php = php.replace(/<\?php[\s\S]*?\?>/g, '');
fs.writeFileSync(path.join(out, 'index.html'), php);

// sw.js: same edits as the Makefile target
var sw = fs.readFileSync(path.join(root, 'dist', 'sw.js'), 'utf8')
	.replace("'timer.php'", "'index.html'")
	.replace(/^\s*'js\/twisty\.js',\r?\n/m, '')
	.replace(/^\s*'js\/cstimer\.js',\r?\n/m, '')
	.replace("'js/jquery.min.js'", "'js/lib/jquery-1.8.0.js'");
var lines = sw.split(/\r?\n/);
lines.pop(); // sed -i '$d'
sw = lines.join('\n') + "\nvar CACHE_NAME = \"cstimer_src_Unspecified\";\n";
fs.writeFileSync(path.join(out, 'sw.js'), sw);

var scripts = [...php.matchAll(/src="js\/([^"]+)"/g)].map(function(m) {
	return m[1];
});
console.log('src_local emulation ready at', out);
console.log('scripts:', scripts.length, '| copilot:',
	scripts.filter(function(s) { return /copilot|alll|scrambleready/.test(s); }).join(','));
