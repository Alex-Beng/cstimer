"use strict";

/**
 * Cube Copilot UI: stage-aware hints for 3x3 CFOP solves.
 *
 * - listens to the live facelet state (bluetooth cube) and to virtual-cube moves
 * - tracks "no progress within the current stage" and offers a hint
 * - renders the hint, and highlights the target slot on the 3D cube
 */
var copilotUI = execMain(function(kernel, timer, tools, cubeutil, logohint) {
	var div = $('<div id="copilotDiv"/>');
	var statusSpan = $('<span class="click"/>');
	var hintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
	var toolHintSpan = null;
	var toolStatusSpan = null;
	var toolChkEnable = null;
	var toolBtnHint = null;
	var toolBtnClear = null;
	var tickTid = 0;
	var hintedFacelet = null;

	var COPILOT_TOOL_HELP_FALLBACK = 'Shows a hint when a stage of a virtual or bluetooth cube solve makes no progress.';
	var COPILOT_COMPUTING_FALLBACK = 'Computing hint...';

	/** language strings fall back to a placeholder when a locale misses them */
	function l10n(value, fallback) {
		return (typeof value == 'string' && value.indexOf('|||') == -1) ? value : fallback;
	}
	var btnHint = $('<input type="button">').val(COPILOT_HINT);
	var btnClear = $('<input type="button">').val(COPILOT_CLEAR);
	var chkEnable = $('<input type="checkbox">');

	var curFacelet = null;
	var curStage = null;
	var stageSince = 0;
	var lastHint = null;
	var isSolving = false;
	var lastWarnAt = 0;

	var workerReady = false;
	var workerBusy = false;
	var reqId = 0;
	var lastSentId = 0;
	var queuedReq = null;

	function isEnabled() {
		return kernel.getProp('cpEnabled', false);
	}

	function timeoutMs() {
		return Math.max(3, ~~kernel.getProp('cpTimeout', 6)) * 1000;
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
		showStatus(l10n(COPILOT_COMPUTING, COPILOT_COMPUTING_FALLBACK));
		setTimeout(function() {
			var hint = computeHintInline(req.facelet, req.manual);
			showHint(hint);
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
			timeLimit: req.timeLimit
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
				showHint(hint);
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
			timeLimit: manual ? 4000 : 1500
		});
	}

	function computeHintInline(facelet, manual) {
		try {
			cubeCopilot.warmUp();
			return cubeCopilot.getHint(facelet, {
				maxDepth: ~~kernel.getProp('cpF2LDepth', 8),
				timeLimit: manual ? 4000 : 1500
			});
		} catch (e) {
			DEBUG && console.log('[copilot] inline hint failed', e);
			return null;
		}
	}

	// ------------------------------------------------------------ state input

	function onFacelet(facelet) {
		if (!facelet || facelet == curFacelet) {
			return;
		}
		curFacelet = facelet;
		hintedFacelet = null;
		lastHint = null;
		hintSpan.empty();
		clearHighlight();
		if (!canRun()) {
			curStage = null;
			return;
		}
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
			return;
		}
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
			hintSpan.empty();
			clearHighlight();
		} else if (status == -1) {
			isSolving = false;
			stageSince = $.now();
			lastWarnAt = 0;
			hintedFacelet = null;
			hintSpan.empty();
			if (toolHintSpan) {
				toolHintSpan.empty();
			}
			clearHighlight();
		}
	}

	// ---------------------------------------------------------------- display

	function renderHint(hint, target) {
		target.empty();
		if (hint.stage == 'f2l') {
			if (hint.name == 'no-solution') {
				target.append(COPILOT_NO_F2L);
				return;
			}
			target.append(
				$('<div/>').html(COPILOT_F2L.replace('%s', hint.name)),
				$('<div/>').css({'font-size': '1.3em'}).text(hint.alg)
			);
		} else if (hint.stage == 'll') {
			if (hint.name == 'no-alg') {
				target.append(COPILOT_NO_ALLL);
				return;
			}
			target.append(
				$('<div/>').html(COPILOT_ALLL.replace('%s', hint.name)),
				$('<div/>').css({'font-size': '1.3em'}).text(hint.alg)
			);
		} else if (hint.stage == 'solved') {
			target.append(COPILOT_SOLVED);
		}
	}

	function showHint(hint) {
		if (!hint) {
			return;
		}
		lastHint = hint;
		renderHint(hint, hintSpan);
		if (toolHintSpan) {
			renderHint(hint, toolHintSpan);
		}
		if (hint.stage == 'f2l' && hint.slot) {
			highlightSlot(hint.slot);
		}
		// Windows are faded out while a solve runs, so the hint is also pushed
		// to the logo banner, which is always visible.
		var toast = hintToast(hint);
		if (toast) {
			logohint.push(toast);
		}
		// replace the transient "computing" status with the normal one
		refreshStatus();
	}

	function hintToast(hint) {
		if (hint.stage == 'f2l') {
			if (hint.name == 'no-solution') {
				return COPILOT_NO_F2L;
			}
			return 'F2L ' + hint.name + ': ' + hint.alg;
		}
		if (hint.stage == 'll') {
			if (hint.name == 'no-alg') {
				return COPILOT_NO_ALLL;
			}
			return COPILOT_ALLL.replace('%s', hint.name) + ' ' + hint.alg;
		}
		return null;
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
		statusSpan.empty();
		if (toolStatusSpan) {
			toolStatusSpan.empty();
		}
		var text = isEnabled() ? COPILOT_WAITING : COPILOT_DISABLED;
		statusSpan.append(text);
		if (toolStatusSpan) {
			toolStatusSpan.append(text);
		}
		chkEnable.prop('checked', isEnabled());
		if (toolChkEnable) {
			toolChkEnable.prop('checked', isEnabled());
		}
		if (isEnabled()) {
			btnHint.show();
			btnClear.show();
			if (toolBtnHint) {
				toolBtnHint.show();
				toolBtnClear.show();
			}
		} else {
			btnHint.hide();
			btnClear.hide();
			if (toolBtnHint) {
				toolBtnHint.hide();
				toolBtnClear.hide();
			}
		}
	}

	function clearAll() {
		curFacelet = null;
		curStage = null;
		lastHint = null;
		hintedFacelet = null;
		stageSince = $.now();
		lastWarnAt = 0;
		hintSpan.empty();
		if (toolHintSpan) {
			toolHintSpan.empty();
		}
		clearHighlight();
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
		toolBtnHint = $('<input type="button">').val(COPILOT_HINT);
		toolBtnClear = $('<input type="button">').val(COPILOT_CLEAR);
		toolHintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
		fdiv.append(
			$('<div/>').css({'font-size': '0.85em', 'margin-bottom': '0.4em'}).html(l10n(COPILOT_TOOL_HELP, COPILOT_TOOL_HELP_FALLBACK)),
			$('<div/>').append($('<label>').append(toolChkEnable, ' ' + COPILOT_ENABLED)),
			toolStatusSpan,
			$('<div/>').append(toolBtnHint, ' ', toolBtnClear),
			toolHintSpan
		);
		toolChkEnable.change(function() {
			kernel.setProp('cpEnabled', this.checked);
		});
		toolBtnHint.click(function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			} else {
				logohint.push(COPILOT_NO_STATE);
			}
		});
		toolBtnClear.click(clearAll);
		if (lastHint) {
			renderHint(lastHint, toolHintSpan);
		}
		refreshStatus();
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
			if (/^(cpEnabled|cpTimeout|cpF2LDepth)$/.exec(value[0])) {
				clearAll();
				refreshStatus();
				if (isEnabled()) {
					startPolling();
				} else {
					stopPolling();
				}
			}
		} else if (signal == 'scramble' || signal == 'time') {
			clearAll();
		}
	}

	$(function() {
		div.append(
			$('<div class="title"/>').append(COPILOT_TITLE),
			$('<div/>').append($('<label>').append(chkEnable, ' ' + COPILOT_ENABLED)),
			statusSpan,
			$('<div/>').append(btnHint, ' ', btnClear),
			hintSpan
		);
		chkEnable.change(function() {
			kernel.setProp('cpEnabled', this.checked);
		});
		btnHint.click(function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			}
		});
		btnClear.click(clearAll);

		kernel.regListener('copilot', 'giikerState', procSignal);
		kernel.regListener('copilot', 'virtualState', procSignal);
		kernel.regListener('copilot', 'timerStatus', procSignal);
		kernel.regListener('copilot', 'property', procSignal, /^(cpEnabled|cpTimeout|cpF2LDepth)$/);
		kernel.regListener('copilot', 'scramble', procSignal);
		kernel.regListener('copilot', 'time', procSignal);

		kernel.regProp('copilot', 'cpEnabled', 0, COPILOT_ENABLED, [false], 1);
		kernel.regProp('copilot', 'cpTimeout', 2, COPILOT_TIMEOUT, [6, 3, 120], 1);
		kernel.regProp('copilot', 'cpF2LDepth', 2, COPILOT_F2LDEPTH, [8, 4, 12], 1);

		kernel.addWindow('copilot', COPILOT_BUTTON, div, false, false, 5);
		tools.regTool('copilot', l10n(TOOLS_COPILOT, 'Copilot'), toolExec);
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
