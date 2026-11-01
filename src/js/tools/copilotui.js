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
	var lastGiikerFacelet = null;
	var isSmart = false;
	var smartOri = null;
	var lockedOri = null;

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
	var hintTracks = null;
	var hintLastSig = null;
	var hintUndo = [];
	var offTrack = false;
	var prevRaw = null;
	var curRawFacelet = null;

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
		return Math.min(120, Math.max(0, ~~kernel.getProp('cpTimeout', 6))) * 1000;
	}


	function llMode() {
		return kernel.getProp('cpLlmode', 'op');
	}

	function f2lCount() {
		return Math.min(4, Math.max(1, ~~kernel.getProp('cpF2LCount', 2)));
	}

	/** smart-cube frame: cross at D, front from a fixed colour order */
	function smartFacelet(raw) {
		// once the cross plus one F2L pair are done the frame is fixed: later
		// algorithms may break the cross temporarily, which must not rotate the
		// hint (or the VRC view) again
		if (lockedOri != null) {
			var f = cubeCopilot.orientFacelet(raw, lockedOri);
			smartOri = {d: f.charAt(31), f: f.charAt(22)};
			if (timer.giiker && timer.giiker.setVrcOri) {
				try {
					timer.giiker.setVrcOri(lockedOri);
				} catch (e) {}
			}
			return f;
		}
		var ori = cubeCopilot.pickSmartOrientation(raw);
		smartOri = {d: ori.d, f: ori.f};
		if (ori.cross && ori.pairCount >= 1) {
			lockedOri = ori.ori;
		}
		// keep the smart-cube VRC view in the same frame as the hint; setOri is
		// a no-op when the view already is at that orientation
		if (ori.cross && timer.giiker && timer.giiker.setVrcOri) {
			try {
				timer.giiker.setVrcOri(ori.ori);
			} catch (e) {}
		}
		return ori.facelet;
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
			showHint(hint, req.facelet, req.raw);
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
				showHint(hint, req.facelet, req.raw);
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
			raw: curRawFacelet,
			manual: !!manual,
			maxDepth: ~~kernel.getProp('cpF2LDepth', 8),
			timeLimit: manual ? 4000 : 1500,
			f2lCount: f2lCount(),
			llMode: llMode()
		});
	}

	function computeHintInline(facelet, manual) {
		try {
			cubeCopilot.warmUp();
			return cubeCopilot.getHint(facelet, {
				maxDepth: ~~kernel.getProp('cpF2LDepth', 8),
				timeLimit: manual ? 4000 : 1500,
				f2lCount: f2lCount(),
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
		var src = (hint.pairs && hint.pairs.length) ? hint.pairs : [{
			name: hint.name,
			key: hint.key,
			slot: hint.slot,
			pair: hint.pair,
			pre: hint.pre,
			post: hint.post,
			base: hint.base,
			moves: hint.moves
		}];
		var pairs = [];
		var canUseCustom = hint.stage != 'solved' && hint.kind != 'skip' && hint.kind != 'off';
		for (var i = 0; i < src.length; i++) {
			var p = src[i];
			var moves = (p.moves || []).slice(0);
			var custom = canUseCustom ? getCustom(p.key) : null;
			if (custom) {
				var alt = tokensOf(p.pre).concat(tokensOf(custom), tokensOf(p.post));
				if (alt.length) {
					moves = alt;
				}
			}
			try {
				moves = cubeCopilot.simplifySeq(moves);
			} catch (e) {}
			pairs.push({
				name: p.name,
				key: p.key,
				slot: p.slot,
				pair: p.pair,
				moves: moves,
				remain: moves.slice(0),
				base: (p.base && p.base.join) ? p.base : moves
			});
		}
		ret.pairs = pairs;
		ret.moves = pairs.length ? pairs[0].moves : (hint.moves || []);
		ret.alg = ret.moves.join(' ');
		ret.remain = pairs.length ? pairs[0].remain : [];
		ret.isCustom = !!(pairs.length && pairs[0].moves.join(' ') != ((src[0] && src[0].moves) || []).join(' '));
		return ret;
	}

	function finishHint() {
		baseHint = null;
		hintInfo = null;
		hintTracks = null;
		hintLastSig = null;
		hintUndo = [];
		offTrack = false;
		prevRaw = null;
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

	function onFacelet(facelet, raw) {
		if (!facelet || facelet == curFacelet) {
			return;
		}
		curFacelet = facelet;
		curRawFacelet = raw || facelet;
		if (!canRun()) {
			curStage = null;
			finishHint();
			return;
		}

		// keep an active hint alive while the user follows it
		if (hintInfo && hintTracks && hintTracks.length) {
			// off the algorithm: show the inverse moves that lead back to it
			if (offTrack) {
				var utok = inferFromPrev(raw);
				if (utok) {
					if (hintUndo.length && utok == hintUndo[hintUndo.length - 1]) {
						hintUndo.pop();
					} else {
						hintUndo.push(cubeCopilot.invertToken(utok));
					}
					prevRaw = raw;
					hintInfo.undo = hintUndo.slice(0);
					if (!hintUndo.length) {
						offTrack = false;
					}
					renderCurrent();
					updateStage(facelet, false);
					return;
				}
				finishHint();
				updateStage(facelet, true);
				requestNext(facelet);
				return;
			}

			var sig = null;
			try {
				sig = cubeCopilot.canonicalSig(facelet);
			} catch (e) {}
			var matched = false;
			var allDone = true;
			for (var t = 0; t < hintTracks.length; t++) {
				var track = hintTracks[t];
				if (track.done) {
					continue;
				}
				var idx = -1;
				if (sig) {
					for (var i = track.cursor; i < track.steps.length; i++) {
						if (track.steps[i].sig == sig) {
							idx = i;
							break;
						}
					}
				}
				if (idx >= 0) {
					matched = true;
					track.cursor = idx + 1;
					if (track.cursor >= track.steps.length) {
						track.done = true;
					} else {
						track.pair.remain = track.steps[idx].remain.slice(0);
					}
				}
				if (!track.done) {
					allDone = false;
				}
			}
			if (matched) {
						hintLastSig = sig;
				prevRaw = raw || prevRaw;
				if (allDone) {
					finishHint();
					updateStage(facelet, true);
					requestNext(facelet);
				} else {
					renderCurrent();
					updateStage(facelet, false);
				}
				return;
			}
			// a whole-cube rotation (x/y/z) is pushed as a new facelet but does
			// not change the position, so it must not count as a wrong move
			if (sig && sig == hintLastSig) {
				prevRaw = raw || prevRaw;
				updateStage(facelet, false);
				return;
			}
			// the user left the algorithm: say which moves undo the deviation
			var bad = inferFromPrev(raw);
			if (bad) {
				offTrack = true;
				hintUndo = [cubeCopilot.invertToken(bad)];
				hintInfo.undo = hintUndo.slice(0);
				prevRaw = raw;
				renderCurrent();
				updateStage(facelet, false);
				return;
			}
			// the move cannot be identified: fall back to a fresh hint
			finishHint();
			updateStage(facelet, true);
			requestNext(facelet);
			return;
		}

		prevRaw = raw || prevRaw;
		updateStage(facelet, false);
		checkTimeout();
	}

	function inferFromPrev(raw) {
		if (!raw || !prevRaw) {
			return null;
		}
		try {
			return cubeCopilot.inferMove(prevRaw, raw);
		} catch (e) {
			return null;
		}
	}

	/** ask for a fresh hint for the current position right away */
	function requestNext(facelet) {
		if (!facelet || !canRun()) {
			return;
		}
		var stage;
		try {
			stage = cubeCopilot.getStage(facelet);
		} catch (e) {
			return;
		}
		if (stage != 'f2l' && stage != 'll') {
			return;
		}
		if (stage == 'll' && llMode() == 'off') {
			return;
		}
		// suppress the poller until this request is answered
		hintedFacelet = facelet;
		lastWarnAt = $.now();
		stageSince = $.now();
		requestHint(facelet, false);
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
			lockedOri = null;
			finishHint();
		} else if (status == -1) {
			isSolving = false;
			stageSince = $.now();
			lastWarnAt = 0;
			hintedFacelet = null;
			lockedOri = null;
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

	function colorNames() {
		var fallback = 'White|Red|Green|Yellow|Orange|Blue';
		var arr = L('COPILOT_COLORS', fallback).split('|');
		if (arr.length != 6) {
			arr = fallback.split('|');
		}
		return arr;
	}

	/** the frame the hint is written in (smart cubes only) */
	function orientationLine() {
		if (!isSmart || !smartOri) {
			return null;
		}
		var names = colorNames();
		var d = names['URFDLB'.indexOf(smartOri.d)] || smartOri.d;
		var f = names['URFDLB'.indexOf(smartOri.f)] || smartOri.f;
		return L('COPILOT_ORI', '%s bottom, %s front').replace('%s', d).replace('%s', f);
	}

	function activePairs(hint) {
		var ret = [];
		var pairs = hint.pairs || [];
		for (var i = 0; i < pairs.length; i++) {
			if (!pairs[i].done && pairs[i].remain && pairs[i].remain.length) {
				ret.push(pairs[i]);
			}
		}
		return ret;
	}

	function undoLine(hint) {
		if (!hint || !hint.undo || !hint.undo.length) {
			return null;
		}
		return hint.undo.slice().reverse().join(' ');
	}

	function renderHint(hint, target) {
		target.empty();
		var msg = hintMessage(hint);
		if (msg) {
			target.append(msg);
			return;
		}
		var ori = orientationLine();
		if (ori) {
			target.append($('<div/>').css({'font-size': '0.85em', 'opacity': 0.85}).text(ori));
		}
		var undo = undoLine(hint);
		if (undo) {
			target.append(
				$('<div/>').html(L('COPILOT_UNDO', 'Off the algorithm, undo: %s').replace('%s', undo))
			);
			return;
		}
		var active = activePairs(hint);
		if (!active.length) {
			target.append(L('COPILOT_SOLVED', 'Solved'));
			return;
		}
		var total = hint.pairs ? hint.pairs.length : 1;
		if (hint.stage == 'f2l' && total > 1) {
			target.append($('<div/>').html(L('COPILOT_F2L_PAIRS', 'Next F2L pairs:')));
			for (var i = 0; i < active.length; i++) {
				target.append(
					$('<div/>').append(
						$('<span/>').text(active[i].name + ': '),
						$('<span/>').css({'font-size': '1.15em'}).text(active[i].remain.join(' '))
					)
				);
			}
			return;
		}
		var pair = active[0];
		var cat = hint.stage == 'f2l' ? L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', pair.name) : hintCategory(hint);
		target.append(
			$('<div/>').html(cat),
			$('<div/>').css({'font-size': '1.3em'}).text(pair.remain.join(' '))
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
		var active = activePairs(hint);
		if (!active.length) {
			hintOverlay.hide().empty();
			return;
		}
		hintOverlay.empty();
		var ori = orientationLine();
		if (ori) {
			hintOverlay.append($('<div class="cp-ori"/>').text(ori));
		}
		var undo = undoLine(hint);
		if (undo) {
			hintOverlay.append($('<div class="cp-cat"/>').html(L('COPILOT_UNDO', 'Off the algorithm, undo: %s').replace('%s', '')));
			hintOverlay.append($('<div class="cp-alg"/>').text(undo));
			hintOverlay.show();
			return;
		}
		var total = hint.pairs ? hint.pairs.length : 1;
		if (hint.stage == 'f2l' && total > 1) {
			hintOverlay.append($('<div class="cp-cat"/>').html(L('COPILOT_F2L_PAIRS', 'Next F2L pairs:')));
			for (var i = 0; i < active.length; i++) {
				hintOverlay.append(
					$('<div class="cp-pair"/>').append(
						$('<span class="cp-name"/>').text(active[i].name + ': '),
						$('<span class="cp-alg2"/>').text(active[i].remain.join(' '))
					)
				);
			}
		} else {
			hintOverlay.append(
				$('<div class="cp-cat"/>').html(hint.stage == 'f2l' ?
					L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', active[0].name) : hintCategory(hint)),
				$('<div class="cp-alg"/>').text(active[0].remain.join(' '))
			);
		}
		hintOverlay.show();
	}

	function hideOverlay() {
		hintOverlay.hide().empty();
	}

	function highlightSlots(slots) {
		try {
			if (timer.virtual && timer.virtual.highlight) {
				timer.virtual.highlight(slots && slots.length ? slots : null);
			}
		} catch (e) {}
	}

	function renderCurrent() {
		if (!hintInfo) {
			return;
		}
		renderHint(hintInfo, hintSpan);
		if (toolHintSpan) {
			renderHint(hintInfo, toolHintSpan);
		}
		var slots = [];
		if (hintInfo.stage == 'f2l' && hintInfo.pairs) {
			for (var i = 0; i < hintInfo.pairs.length; i++) {
				var p = hintInfo.pairs[i];
				if (!p.done && p.slot) {
					slots = slots.concat(p.slot);
				}
			}
		}
		highlightSlots(slots);
		showOverlay(hintInfo);
	}

	function buildTracks(facelet, info) {
		hintTracks = [];
		if (!info || !info.pairs) {
			return;
		}
		for (var i = 0; i < info.pairs.length; i++) {
			var p = info.pairs[i];
			if (!p || !p.moves || !p.moves.length) {
				continue;
			}
			var steps = null;
			try {
				steps = cubeCopilot.algSteps(facelet, p.moves);
			} catch (e) {}
			if (steps) {
				hintTracks.push({steps: steps, cursor: 0, done: false, pair: p});
			}
		}
	}

	function showHint(hint, facelet, raw) {
		if (!hint) {
			return;
		}
		baseHint = hint;
		hintInfo = composeHint(hint);
		hintFacelet = facelet;
		hintLastSig = null;
		hintUndo = [];
		offTrack = false;
		hintInfo.undo = [];
		prevRaw = raw || null;
		if (facelet) {
			try {
				hintLastSig = cubeCopilot.canonicalSig(facelet);
			} catch (e) {}
		}
		buildTracks(facelet, hintInfo);
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
		lockedOri = null;
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
		hintInfo.undo = [];
		hintLastSig = null;
		hintUndo = [];
		offTrack = false;
		if (hintFacelet) {
			try {
				hintLastSig = cubeCopilot.canonicalSig(hintFacelet);
			} catch (e) {}
		}
		buildTracks(hintFacelet, hintInfo);
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
			isSmart = true;
			lastGiikerFacelet = value[0];
			onFacelet(smartFacelet(value[0]), value[0]);
		} else if (signal == 'virtualState') {
			isSmart = false;
			smartOri = null;
			onFacelet(value[0], value[0]);
		} else if (signal == 'timerStatus') {
			onTimerStatus(value);
		} else if (signal == 'property') {
			if (/^(cpEnabled|cpTimeout|cpF2LDepth|cpLlmode|cpF2LCount)$/.exec(value[0])) {
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
		kernel.regProp('copilot', 'cpTimeout', 2, L('COPILOT_TIMEOUT', 'Hint after no progress (s)'), [6, 0, 120], 1);
		kernel.regProp('copilot', 'cpF2LDepth', 2, L('COPILOT_F2LDEPTH', 'F2L max search depth'), [8, 4, 12], 1);
		kernel.regProp('copilot', 'cpF2LCount', 2, L('COPILOT_F2LCOUNT', 'Number of F2L hints'), [2, 1, 4], 1);
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
		kernel.regListener('copilot', 'property', procSignal, /^(cpEnabled|cpTimeout|cpF2LDepth|cpLlmode|cpF2LCount|cpCustom)$/);
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
