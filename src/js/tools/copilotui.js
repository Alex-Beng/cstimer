"use strict";

/**
 * Cube Copilot UI: stage-aware hints for 3x3 CFOP solves.
 *
 * - listens to the live facelet state (bluetooth cube) and to virtual-cube moves
 * - tracks "no progress within the current stage" and offers a hint
 * - keeps a shown hint alive while the moves follow it, and drops it after a
 *   configurable number of non-matching moves
 * - renders the hint in a fixed overlay (visible while solving), highlights the
 *   target slot, and lets every case carry a user-supplied algorithm
 */
var copilotUI = execMain(function(kernel, timer, tools, cubeutil, logohint) {
	var div = $('<div id="copilotDiv"/>');
	var statusSpan = $('<span class="click"/>');
	var hintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
	var hintOverlay = $('<div id="copilotHint"/>');
	var toolHintSpan = null;
	var toolStatusSpan = null;
	var toolChkEnable = null;
	var toolBtnHint = null;
	var toolBtnClear = null;
	var tickTid = 0;
	var hintedFacelet = null;

	var btnHint = $('<input type="button">').val(L('COPILOT_HINT', 'Hint'));
	var btnClear = $('<input type="button">').val(L('COPILOT_CLEAR', 'Clear'));
	var chkEnable = $('<input type="checkbox">');

	var curFacelet = null;
	var curStage = null;
	var stageSince = 0;
	var lastWarnAt = 0;
	var isSolving = false;

	/* active hint */
	var baseHint = null;
	var hintInfo = null;
	var hintFacelet = null;
	var hintSig = null;
	var hintCursor = 0;
	var hintMismatch = 0;

	var customRows = [];

	var workerReady = false;
	var workerBusy = false;
	var reqId = 0;
	var lastSentId = 0;
	var queuedReq = null;

	/**
	 * Language strings live as globals of the selected locale file, and most
	 * locales predate this module, so every lookup must tolerate a missing
	 * string instead of throwing a ReferenceError.
	 */
	function L(name, fallback) {
		try {
			var v = (typeof window != 'undefined') ? window[name] : undefined;
			return (typeof v == 'string' && v.indexOf('|||') == -1) ? v : fallback;
		} catch (e) {
			return fallback;
		}
	}

	function isEnabled() {
		return kernel.getProp('cpEnabled', false);
	}

	function timeoutMs() {
		return Math.min(120, Math.max(1, ~~kernel.getProp('cpTimeout', 6))) * 1000;
	}

	function driftMoves() {
		return Math.min(20, Math.max(1, ~~kernel.getProp('cpDrift', 2)));
	}

	function llMode() {
		return kernel.getProp('cpLlmode', 'op');
	}

	function isCube3() {
		return tools.getCurPuzzle() == '333';
	}

	function canRun() {
		if (!isEnabled() || !isCube3()) {
			return false;
		}
		if (tools.isCurTrainScramble && tools.isCurTrainScramble()) {
			return false;
		}
		return true;
	}

	function tokensOf(str) {
		return str ? String(str).split(/\s+/).filter(function(t) {
			return !!t;
		}) : [];
	}

	// ------------------------------------------------------------ custom algs

	function getCustomMap() {
		var raw = kernel.getProp('cpCustom', '');
		if (!raw) {
			return {};
		}
		try {
			var obj = (typeof raw == 'string') ? JSON.parse(raw) : raw;
			return (obj && typeof obj == 'object') ? obj : {};
		} catch (e) {
			return {};
		}
	}

	function getCustom(key) {
		if (!key) {
			return null;
		}
		var v = getCustomMap()[key];
		return (typeof v == 'string' && v.length) ? v : null;
	}

	function setCustom(key, alg) {
		if (!key) {
			return;
		}
		var map = getCustomMap();
		if (alg) {
			map[key] = alg;
		} else {
			delete map[key];
		}
		kernel.setProp('cpCustom', JSON.stringify(map));
	}

	/**
	 * Accept the notation people actually type: lowercase wide turns (r, u, ...)
	 * as well as csTimer's Rw form, plus slices and rotations.
	 */
	function parseUserAlg(text) {
		var raw = String(text || '').replace(/[()\[\]]/g, ' ');
		var out = [];
		var i = 0;
		while (i < raw.length) {
			var ch = raw.charAt(i);
			if (/\s|,/.test(ch)) {
				i++;
				continue;
			}
			if ('URFDLBurfdlbMESxyz'.indexOf(ch) == -1) {
				return null;
			}
			var base = ch;
			i++;
			var wide = '';
			if (raw.charAt(i) == 'w') {
				wide = 'w';
				i++;
			}
			var mod = '';
			if (raw.charAt(i) == '2') {
				mod = '2';
				i++;
				if (raw.charAt(i) == "'") {
					i++;
				}
			} else if (raw.charAt(i) == "'") {
				mod = "'";
				i++;
			}
			if ('urfdlb'.indexOf(base) != -1) {
				base = base.toUpperCase();
				wide = 'w';
			}
			out.push(base + wide + mod);
		}
		return out.length ? out : null;
	}

	// ------------------------------------------------------------ worker glue

	function workerUsable() {
		if (!csTimerWorker || !csTimerWorker.getHint) {
			return false;
		}
		if (csTimerWorker.isAvailable && !csTimerWorker.isAvailable()) {
			return false;
		}
		return true;
	}

	function initWorker() {
		if (workerReady || !workerUsable()) {
			return;
		}
		workerReady = true;
		if (csTimerWorker.warmUp) {
			csTimerWorker.warmUp();
		}
	}

	/**
	 * A source-only build has no js/cstimer.js, so the worker never answers.
	 * The hint is then computed on the main thread, deferred so the click/key
	 * handler is not blocked.
	 */
	function inlineRequest(req) {
		showStatus(L('COPILOT_COMPUTING', 'Computing hint...'));
		setTimeout(function() {
			var hint = computeHintInline(req.facelet, req.manual);
			showHint(hint, req.facelet);
		}, 0);
	}

	function postRequest(req) {
		if (!workerUsable()) {
			inlineRequest(req);
			return;
		}
		if (workerBusy) {
			queuedReq = req;
			return;
		}
		workerBusy = true;
		lastSentId = req.id;
		var answered = false;
		var fallbackTid = setTimeout(function() {
			// the worker accepted the message but never replied
			if (answered) {
				return;
			}
			answered = true;
			workerBusy = false;
			inlineRequest(req);
			if (queuedReq) {
				var q = queuedReq;
				queuedReq = null;
				postRequest(q);
			}
		}, 9000);
		csTimerWorker.getHint(req.facelet, {
			maxDepth: req.maxDepth,
			timeLimit: req.timeLimit,
			llMode: req.llMode
		}, function(hint) {
			if (answered) {
				return;
			}
			answered = true;
			clearTimeout(fallbackTid);
			workerBusy = false;
			if (hint == null) {
				inlineRequest(req);
			} else if (req.id == lastSentId) {
				showHint(hint, req.facelet);
			}
			if (queuedReq) {
				var next = queuedReq;
				queuedReq = null;
				postRequest(next);
			}
		});
	}

	function requestHint(facelet, manual) {
		if (!canRun() || !facelet) {
			return;
		}
		initWorker();
		postRequest({
			id: ++reqId,
			facelet: facelet,
			manual: !!manual,
			maxDepth: ~~kernel.getProp('cpF2LDepth', 8),
			timeLimit: manual ? 4000 : 1500,
			llMode: llMode()
		});
	}

	function computeHintInline(facelet, manual) {
		try {
			cubeCopilot.warmUp();
			return cubeCopilot.getHint(facelet, {
				maxDepth: ~~kernel.getProp('cpF2LDepth', 8),
				timeLimit: manual ? 4000 : 1500,
				llMode: llMode()
			});
		} catch (e) {
			DEBUG && console.log('[copilot] inline hint failed', e);
			return null;
		}
	}

	// ------------------------------------------------------------ state input

	/** apply the user's override (if any) to a hint coming from the engine */
	function composeHint(hint) {
		if (!hint) {
			return hint;
		}
		var ret = {};
		for (var k in hint) {
			ret[k] = hint[k];
		}
		ret.remain = hint.moves ? hint.moves.slice(0) : [];
		ret.isCustom = false;
		var custom = getCustom(hint.key);
		if (custom && hint.stage != 'solved' && hint.kind != 'skip' && hint.kind != 'off') {
			var moves = tokensOf(hint.pre).concat(tokensOf(custom), tokensOf(hint.post));
			if (moves.length) {
				ret.moves = moves;
				ret.alg = moves.join(' ');
				ret.remain = moves.slice(0);
				ret.isCustom = true;
			}
		}
		return ret;
	}

	function finishHint() {
		baseHint = null;
		hintInfo = null;
		hintSig = null;
		hintCursor = 0;
		hintMismatch = 0;
		hintSpan.empty();
		if (toolHintSpan) {
			toolHintSpan.empty();
		}
		hideOverlay();
		clearHighlight();
		syncCustomRows();
	}

	function updateStage(facelet, resetTimer) {
		var stage;
		try {
			stage = cubeCopilot.getStage(facelet);
		} catch (e) {
			return;
		}
		if (stage != curStage) {
			curStage = stage;
			stageSince = $.now();
			lastWarnAt = 0;
		} else if (resetTimer) {
			stageSince = $.now();
		}
	}

	function onFacelet(facelet) {
		if (!facelet || facelet == curFacelet) {
			return;
		}
		curFacelet = facelet;
		if (!canRun()) {
			curStage = null;
			finishHint();
			return;
		}

		// keep an active hint alive while the user follows it
		if (hintInfo && hintSig) {
			var sig = null;
			try {
				sig = cubeCopilot.normalizeFacelet(facelet);
			} catch (e) {}
			var idx = -1;
			if (sig) {
				for (var i = hintCursor; i < hintSig.length; i++) {
					if (hintSig[i].sig == sig) {
						idx = i;
						break;
					}
				}
			}
			if (idx >= 0) {
				hintCursor = idx + 1;
				hintMismatch = 0;
				if (hintCursor >= hintSig.length) {
					finishHint();
					updateStage(facelet, true);
				} else {
					hintInfo.remain = hintSig[idx].remain.slice(0);
					renderCurrent();
					updateStage(facelet, false);
				}
				return;
			}
			hintMismatch++;
			if (hintMismatch < driftMoves()) {
				updateStage(facelet, false);
				return;
			}
			finishHint();
			updateStage(facelet, true);
			return;
		}

		updateStage(facelet, false);
		checkTimeout();
	}

	/**
	 * A stalled solve produces no new facelets, so the timeout cannot rely on
	 * the state callback: it is polled while a solve is running.
	 */
	function checkTimeout() {
		if (!canRun() || !curFacelet || !isSolving) {
			return;
		}
		if (curStage != 'f2l' && curStage != 'll') {
			return;
		}
		// never replace a hint the user is still following
		if (hintInfo) {
			return;
		}
		if (curStage == 'll' && llMode() == 'off') {
			return;
		}
		// one hint per position: do not nag while the user is thinking
		if (curFacelet == hintedFacelet) {
			return;
		}
		var now = $.now();
		if (now - stageSince < timeoutMs()) {
			return;
		}
		if (now - lastWarnAt < timeoutMs()) {
			return;
		}
		lastWarnAt = now;
		hintedFacelet = curFacelet;
		requestHint(curFacelet, false);
	}

	function startPolling() {
		if (tickTid) {
			return;
		}
		tickTid = setInterval(checkTimeout, 500);
	}

	function stopPolling() {
		if (tickTid) {
			clearInterval(tickTid);
			tickTid = 0;
		}
	}

	function onTimerStatus(status) {
		if (status >= 1) {
			isSolving = true;
			stageSince = $.now();
			lastWarnAt = 0;
			finishHint();
		} else if (status == -1) {
			isSolving = false;
			stageSince = $.now();
			lastWarnAt = 0;
			hintedFacelet = null;
			finishHint();
		}
	}

	// ---------------------------------------------------------------- display

	function hintCategory(hint) {
		if (hint.stage == 'f2l') {
			return L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', hint.name);
		}
		if (hint.kind == 'oll') {
			return L('COPILOT_OLL', 'OLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'pll') {
			return L('COPILOT_PLL', 'PLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'auf') {
			return L('COPILOT_AUF', 'Adjust the top layer');
		}
		return L('COPILOT_ALLL', 'One-look last layer: %s').replace('%s', hint.name);
	}

	function hintMessage(hint) {
		if (!hint) {
			return L('COPILOT_NO_F2L', 'No F2L hint found');
		}
		if (hint.stage == 'solved' || hint.kind == 'skip') {
			return L('COPILOT_SOLVED', 'Solved');
		}
		if (hint.kind == 'off') {
			return L('COPILOT_LL_OFF_MSG', 'Last-layer hints are off');
		}
		if (hint.name == 'no-solution') {
			return L('COPILOT_NO_F2L', 'No F2L hint found');
		}
		if (hint.name == 'no-alg') {
			return L('COPILOT_NO_ALLL', 'No algorithm for this last-layer case');
		}
		return null;
	}

	function renderHint(hint, target) {
		target.empty();
		var msg = hintMessage(hint);
		if (msg) {
			target.append(msg);
			return;
		}
		var alg = hint.remain && hint.remain.length ? hint.remain.join(' ') : hint.alg;
		target.append(
			$('<div/>').html(hintCategory(hint)),
			$('<div/>').css({'font-size': '1.3em'}).text(alg)
		);
	}

	/**
	 * During a solve csTimer fades #leftbar to 0.01 opacity (taking the logo
	 * banner with it) and auto-hides module windows, so the hint needs its own
	 * fixed overlay to be visible.
	 */
	function showOverlay(hint) {
		var msg = hintMessage(hint);
		if (msg) {
			hintOverlay.hide().empty();
			return;
		}
		var alg = hint.remain && hint.remain.length ? hint.remain.join(' ') : hint.alg;
		if (!alg) {
			hintOverlay.hide().empty();
			return;
		}
		hintOverlay.empty().append(
			$('<div class="cp-cat"/>').html(hintCategory(hint)),
			$('<div class="cp-alg"/>').text(alg)
		).show();
	}

	function hideOverlay() {
		hintOverlay.hide().empty();
	}

	function renderCurrent() {
		if (!hintInfo) {
			return;
		}
		renderHint(hintInfo, hintSpan);
		if (toolHintSpan) {
			renderHint(hintInfo, toolHintSpan);
		}
		if (hintInfo.stage == 'f2l' && hintInfo.slot) {
			highlightSlot(hintInfo.slot);
		} else {
			clearHighlight();
		}
		showOverlay(hintInfo);
	}

	function showHint(hint, facelet) {
		if (!hint) {
			return;
		}
		baseHint = hint;
		hintInfo = composeHint(hint);
		hintFacelet = facelet;
		hintCursor = 0;
		hintMismatch = 0;
		hintSig = null;
		if (hintInfo.moves && hintInfo.moves.length) {
			try {
				hintSig = cubeCopilot.algSteps(facelet, hintInfo.moves);
			} catch (e) {}
		}
		// remember this position so the poller does not ask again for it
		if (facelet) {
			hintedFacelet = facelet;
		}
		renderCurrent();
		syncCustomRows();
		// replace the transient "computing" status with the normal one
		refreshStatus();
	}

	function clearHighlight() {
		try {
			if (timer.virtual && timer.virtual.highlight) {
				timer.virtual.highlight(null);
			}
		} catch (e) {}
	}

	function highlightSlot(slot) {
		try {
			if (timer.virtual && timer.virtual.highlight) {
				timer.virtual.highlight(slot);
			}
		} catch (e) {}
	}

	function showStatus(text) {
		statusSpan.empty().append(text);
		if (toolStatusSpan) {
			toolStatusSpan.empty().append(text);
		}
	}

	function refreshStatus() {
		chkEnable.prop('checked', isEnabled());
		if (toolChkEnable) {
			toolChkEnable.prop('checked', isEnabled());
		}
		showStatus(isEnabled() ? L('COPILOT_WAITING', 'Waiting for a solve...') :
			L('COPILOT_DISABLED', 'Copilot is off'));
		var show = isEnabled();
		btnHint.toggle(show);
		btnClear.toggle(show);
		if (toolBtnHint) {
			toolBtnHint.toggle(show);
			toolBtnClear.toggle(show);
		}
	}

	function clearAll() {
		curFacelet = null;
		curStage = null;
		stageSince = $.now();
		lastWarnAt = 0;
		hintedFacelet = null;
		finishHint();
	}

	// ------------------------------------------------------ custom alg editor

	function customRowLabel(hint) {
		var title = L('COPILOT_CUSTOM', 'Custom algorithm');
		if (!hint) {
			return title;
		}
		return title + ': ' + hintCategory(hint);
	}

	function syncCustomRows() {
		var key = (hintInfo && hintInfo.key) ? hintInfo.key : '';
		var val = key ? (getCustom(key) || '') : '';
		var label = customRowLabel(hintInfo);
		for (var i = 0; i < customRows.length; i++) {
			var row = customRows[i];
			row.wrap.toggle(!!key);
			row.label.text(label);
			if (document.activeElement != row.input[0]) {
				row.input.val(val);
			}
			row.reset.toggle(!!val);
		}
	}

	function rebuildHintFromDefault() {
		if (!baseHint) {
			syncCustomRows();
			return;
		}
		hintInfo = composeHint(baseHint);
		hintCursor = 0;
		hintMismatch = 0;
		hintSig = null;
		if (hintFacelet && hintInfo.moves && hintInfo.moves.length) {
			try {
				hintSig = cubeCopilot.algSteps(hintFacelet, hintInfo.moves);
			} catch (e) {}
		}
		renderCurrent();
		syncCustomRows();
	}

	function saveCustomFrom(row) {
		if (!hintInfo || !hintFacelet) {
			return;
		}
		var edited = parseUserAlg(row.input.val());
		if (!edited) {
			showStatus(L('COPILOT_BAD_ALG', 'Invalid algorithm'));
			return;
		}
		var full = tokensOf(hintInfo.pre).concat(edited, tokensOf(hintInfo.post));
		var ok = false;
		try {
			ok = cubeCopilot.validateAlg(hintFacelet, full, hintInfo);
		} catch (e) {}
		if (!ok) {
			showStatus(L('COPILOT_CUSTOM_BAD', 'That algorithm does not solve this case'));
			return;
		}
		setCustom(hintInfo.key, edited.join(' '));
		rebuildHintFromDefault();
	}

	function resetCustomFrom() {
		if (!hintInfo) {
			return;
		}
		setCustom(hintInfo.key, '');
		rebuildHintFromDefault();
	}

	function makeCustomRow() {
		var label = $('<div/>').css({'font-size': '0.85em', 'margin-top': '0.4em'});
		var input = $('<input type="text">').css({width: '100%'});
		var save = $('<input type="button">').val(L('COPILOT_SAVE', 'Save'));
		var reset = $('<input type="button">').val(L('COPILOT_RESET', 'Reset'));
		var wrap = $('<div/>').append(label, input, $('<div/>').append(save, ' ', reset)).hide();
		var row = {wrap: wrap, label: label, input: input, save: save, reset: reset};
		customRows.push(row);
		save.click(function() {
			saveCustomFrom(row);
		});
		reset.click(function() {
			resetCustomFrom();
		});
		input.keydown(function(e) {
			if (e.which == 13) {
				saveCustomFrom(row);
			}
		});
		return wrap;
	}

	/**
	 * Tools menu entry: same state as the window, so the feature is also
	 * reachable from Tools instead of only the left-bar window button.
	 */
	function toolExec(fdiv, signal) {
		if (fdiv === undefined) {
			toolHintSpan = toolStatusSpan = null;
			toolChkEnable = toolBtnHint = toolBtnClear = null;
			return;
		}
		fdiv.empty();
		toolStatusSpan = $('<div/>');
		toolChkEnable = $('<input type="checkbox">');
		toolBtnHint = $('<input type="button">').val(L('COPILOT_HINT', 'Hint'));
		toolBtnClear = $('<input type="button">').val(L('COPILOT_CLEAR', 'Clear'));
		toolHintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
		fdiv.append(
			$('<div/>').css({'font-size': '0.85em', 'margin-bottom': '0.4em'}).html(
				L('COPILOT_TOOL_HELP', 'While solving on a virtual or bluetooth cube, a hint is shown when a stage makes no progress (tune it in Options > Copilot).')),
			$('<div/>').append($('<label>').append(toolChkEnable, ' ' + L('COPILOT_ENABLED', 'Enable Copilot'))),
			toolStatusSpan,
			$('<div/>').append(toolBtnHint, ' ', toolBtnClear),
			makeCustomRow(),
			toolHintSpan
		);
		toolChkEnable.change(function() {
			kernel.setProp('cpEnabled', this.checked);
		});
		toolBtnHint.click(function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			} else {
				logohint.push(L('COPILOT_NO_STATE', 'No cube state yet'));
			}
		});
		toolBtnClear.click(clearAll);
		if (hintInfo) {
			renderHint(hintInfo, toolHintSpan);
		}
		refreshStatus();
		syncCustomRows();
	}

	// ------------------------------------------------------------- signal glue

	function procSignal(signal, value) {
		if (signal == 'giikerState') {
			onFacelet(value[0]);
		} else if (signal == 'virtualState') {
			onFacelet(value[0]);
		} else if (signal == 'timerStatus') {
			onTimerStatus(value);
		} else if (signal == 'property') {
			if (/^(cpEnabled|cpTimeout|cpF2LDepth|cpLlmode|cpDrift)$/.exec(value[0])) {
				clearAll();
				refreshStatus();
				if (isEnabled()) {
					startPolling();
				} else {
					stopPolling();
				}
			} else if (value[0] == 'cpCustom') {
				rebuildHintFromDefault();
			}
		} else if (signal == 'scramble' || signal == 'time') {
			clearAll();
		}
	}

	function initProps() {
		kernel.regProp('copilot', 'cpEnabled', 0, L('COPILOT_ENABLED', 'Enable Copilot'), [false], 1);
		kernel.regProp('copilot', 'cpTimeout', 2, L('COPILOT_TIMEOUT', 'Hint after no progress (s)'), [6, 1, 120], 1);
		kernel.regProp('copilot', 'cpDrift', 2, L('COPILOT_DRIFT', 'Hide hint after non-matching moves'), [2, 1, 20], 1);
		kernel.regProp('copilot', 'cpF2LDepth', 2, L('COPILOT_F2LDEPTH', 'F2L max search depth'), [8, 4, 12], 1);
		kernel.regProp('copilot', 'cpLlmode', 1, L('COPILOT_LLMODE', 'Last-layer method'), ['op', ['op', 'alll', 'off'], [
			L('COPILOT_LL_OP', 'OLL + PLL'),
			L('COPILOT_LL_ALLL', '1LLL (ALLL)'),
			L('COPILOT_LL_OFF', 'Off')
		]], 1);
		kernel.regProp('copilot', 'cpCustom', ~5, 'Copilot custom algorithms', ['{}'], 1);
	}

	$(function() {
		div.append(
			$('<div class="title"/>').append(L('COPILOT_TITLE', 'Virtual/Bluetooth Cube Copilot')),
			$('<div/>').append($('<label>').append(chkEnable, ' ' + L('COPILOT_ENABLED', 'Enable Copilot'))),
			statusSpan,
			$('<div/>').append(btnHint, ' ', btnClear),
			makeCustomRow(),
			hintSpan
		);
		btnHint.click(function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			}
		});
		btnClear.click(clearAll);
		chkEnable.change(function() {
			kernel.setProp('cpEnabled', this.checked);
		});

		kernel.regListener('copilot', 'giikerState', procSignal);
		kernel.regListener('copilot', 'virtualState', procSignal);
		kernel.regListener('copilot', 'timerStatus', procSignal);
		kernel.regListener('copilot', 'property', procSignal, /^(cpEnabled|cpTimeout|cpF2LDepth|cpLlmode|cpDrift|cpCustom)$/);
		kernel.regListener('copilot', 'scramble', procSignal);
		kernel.regListener('copilot', 'time', procSignal);

		initProps();

		kernel.addWindow('copilot', L('COPILOT_BUTTON', 'HINT'), div, false, false, 5);
		tools.regTool('copilot', L('TOOLS_COPILOT', 'Copilot'), toolExec);
		hintOverlay.appendTo('body');
		chkEnable.prop('checked', isEnabled());
		refreshStatus();
		startPolling();
	});

	return {
		onFacelet: onFacelet,
		requestHint: function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			}
		},
		clear: clearAll
	};
}, [kernel, timer, tools, cubeutil, logohint]);
