"use strict";

/**
 * Cube Copilot UI: stage-aware hints for 3x3 CFOP solves.
 *
 * - listens to the live facelet state (bluetooth cube) and to virtual-cube moves
 * - tracks "no progress within the current stage" and offers a hint
 * - renders the hint, and highlights the target slot on the 3D cube
 */
var copilotUI = execMain(function(kernel, timer, tools, cubeutil) {
	var div = $('<div id="copilotDiv"/>');
	var statusSpan = $('<span class="click"/>');
	var hintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
	var btnHint = $('<input type="button">').val(COPILOT_HINT);
	var btnClear = $('<input type="button">').val(COPILOT_CLEAR);

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
		return Math.max(3, ~~kernel.getProp('cpTimeout', 10)) * 1000;
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

	function initWorker() {
		if (workerReady || !csTimerWorker || !csTimerWorker.getHint) {
			return;
		}
		workerReady = true;
		if (csTimerWorker.warmUp) {
			csTimerWorker.warmUp();
		}
	}

	function postRequest(req) {
		if (!csTimerWorker || !csTimerWorker.getHint) {
			showHint(computeHintInline(req.facelet, req.manual));
			return;
		}
		if (workerBusy) {
			queuedReq = req;
			return;
		}
		workerBusy = true;
		lastSentId = req.id;
		csTimerWorker.getHint(req.facelet, {
			maxDepth: req.maxDepth,
			timeLimit: req.timeLimit
		}, function(hint) {
			workerBusy = false;
			if (req.id == lastSentId) {
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
		if (stage == 'solved' || stage == 'cross') {
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
		requestHint(facelet, false);
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
		}
	}

	// ---------------------------------------------------------------- display

	function showHint(hint) {
		if (!hint) {
			return;
		}
		lastHint = hint;
		hintSpan.empty();
		if (hint.stage == 'f2l') {
			if (hint.name == 'no-solution') {
				hintSpan.append(COPILOT_NO_F2L);
				return;
			}
			hintSpan.append(
				$('<div/>').html(COPILOT_F2L.replace('%s', hint.name)),
				$('<div/>').css({'font-size': '1.3em'}).text(hint.alg)
			);
			highlightSlot(hint.slot);
		} else if (hint.stage == 'll') {
			if (hint.name == 'no-alg') {
				hintSpan.append(COPILOT_NO_ALLL);
				return;
			}
			hintSpan.append(
				$('<div/>').html(COPILOT_ALLL.replace('%s', hint.name)),
				$('<div/>').css({'font-size': '1.3em'}).text(hint.alg)
			);
		} else if (hint.stage == 'solved') {
			hintSpan.append(COPILOT_SOLVED);
		}
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

	function refreshStatus() {
		statusSpan.empty();
		if (!isEnabled()) {
			statusSpan.append(COPILOT_DISABLED);
			btnHint.hide();
			btnClear.hide();
			return;
		}
		btnHint.show();
		btnClear.show();
		statusSpan.append(COPILOT_WAITING);
	}

	function clearAll() {
		curFacelet = null;
		curStage = null;
		lastHint = null;
		stageSince = $.now();
		lastWarnAt = 0;
		hintSpan.empty();
		clearHighlight();
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
			}
		} else if (signal == 'scramble' || signal == 'time') {
			clearAll();
		}
	}

	$(function() {
		div.append(
			$('<div class="title"/>').append(COPILOT_TITLE),
			statusSpan,
			$('<div/>').append(btnHint, ' ', btnClear),
			hintSpan
		);
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
		kernel.regProp('copilot', 'cpTimeout', 2, COPILOT_TIMEOUT, [10, 3, 120], 1);
		kernel.regProp('copilot', 'cpF2LDepth', 2, COPILOT_F2LDEPTH, [8, 4, 12], 1);

		kernel.addWindow('copilot', COPILOT_BUTTON, div, false, true, 5);
		refreshStatus();
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
}, [kernel, timer, tools, cubeutil]);
