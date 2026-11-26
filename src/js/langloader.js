/**
 * Language loader for the static GitHub Pages build.
 *
 * The PHP build normally picks a language on the server.  A static host
 * cannot do that, so the generated page always contains English as a
 * fallback and this loader replaces it before the application boots.
 *
 * It is deliberately synchronous: document.write() keeps the locale script
 * in front of kernel.js and the rest of the application.
 */
(function() {
	var codes = (window.LANG_SET || '|en-us').split('|').filter(function(v) {
		return v;
	});
	var baseLang = window.LANG_CUR || 'en-us';

	function normalize(value) {
		if (!value) {
			return '';
		}
		value = String(value).toLowerCase();
		if (value == 'cn') {
			value = 'zh-cn';
		}
		for (var i = 0; i < codes.length; i++) {
			if (value.slice(0, 5) == codes[i]) {
				return codes[i];
			}
		}
		for (i = 0; i < codes.length; i++) {
			if (value.slice(0, 2) == codes[i].slice(0, 2)) {
				return codes[i];
			}
		}
		return '';
	}

	var query = /[?&]lang=([^&]+)/.exec(window.location.search);
	var fromQuery = query ? decodeURIComponent(query[1].replace(/\+/g, ' ')) : '';
	var cookie = /(?:^|;\s*)lang=([^;]+)/.exec(document.cookie || '');
	var fromCookie = cookie ? decodeURIComponent(cookie[1]) : '';
	var candidates = [];
	if (fromQuery) {
		candidates.push(fromQuery);
	}
	if (fromCookie) {
		candidates.push(fromCookie);
	}
	if (navigator.languages && navigator.languages.length) {
		candidates = candidates.concat(navigator.languages);
	} else if (navigator.language) {
		candidates.push(navigator.language);
	}

	var lang = 'en-us';
	for (var i = 0; i < candidates.length; i++) {
		var found = normalize(candidates[i]);
		if (found) {
			lang = found;
			break;
		}
	}

	window.LANG_CUR = lang;
	document.documentElement.lang = lang;

	if (fromQuery) {
		document.cookie = 'lang=' + lang + '; expires=Fri, 31 Dec 9999 23:59:59 GMT; path=/';
	}
	if (lang != baseLang) {
		document.write('<script type="text/javascript" src="lang/' + lang + '.js"><\/script>');
	}
})();
