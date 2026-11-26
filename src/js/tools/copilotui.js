"use strict";

/**
 * Cube Copilot UI: stage-aware hints for 3x3 CFOP and Roux solves.
 *
 * - listens to the live facelet state (bluetooth cube) and to virtual-cube moves
 * - tracks "no progress within the current stage" and offers a hint
 * - keeps a shown hint alive while the moves follow it, and drops it after a
 *   configurable number of non-matching moves
 * - renders the hint in a fixed overlay (visible while solving), highlights the
 *   F2L corner/edge pieces that need the fewest moves, and lets every
 *   last-layer (or CMLL) case carry a user-supplied algorithm
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
	var onTrackPairs = 0;
	/* live F2L highlight: the pair with the fewest moves, no algorithm */
	var f2lPair = null;
	/* first/second-layer hint: highlight follows the piece it targets */
	var lblPiece = null;

	var customRows = [];

	var workerReady = false;
	var workerBusy = false;
	var reqId = 0;
	var lastSentId = 0;
	var queuedReq = null;

	/* 3x3 relay pacing feedback: breathing edge glow + one small flash per cube */
	var relayGlow = $('<div id="copilotRelayGlow"/>');
	var relayFlash = $('<div id="copilotRelayFlash"/>');
	var relayFinished = false; // the last cube is done, keep its colour
	var relayClearTid = 0; // clears the final colour a moment after the last cube

	/**
	 * The relay overlay styles ship with this script instead of style.css: the
	 * service worker precaches style.css but not the JS modules, so a stale
	 * cached stylesheet used to leave the overlay without any styling.  An
	 * explicit breathing opacity also keeps it visible when animations are off.
	 */
	var relayStyle = [
		'#copilotRelayGlow, #copilotRelayFlash {',
		'    position: fixed;',
		'    left: 0;',
		'    top: 0;',
		'    width: 100%;',
		'    height: 100%;',
		'    pointer-events: none;',
		'    opacity: 0;',
		'    background-repeat: no-repeat;',
		'}',
		'#copilotRelayGlow {',
		'    z-index: 40;',
		/* depth knobs: a slower cumulative pace makes the red deeper and wider */
		'    --relay-ce: 205, 0, 0;',
		'    --relay-cm: 165, 0, 0;',
		'    --relay-ct: 130, 0, 0;',
		'    --relay-ck: 110, 0, 0;',
		'    --relay-ey: 17vmin;',
		'    --relay-ex: 20vmin;',
		'    --relay-cc: 32vmin;',
		'}',
		'#copilotRelayFlash { z-index: 41; }',
		/* four gradient strips + darker corners: only the screen edges tint */
		'#copilotRelayGlow.relay-glow-ok {',
		'    background-image:',
		'        linear-gradient(to bottom, rgba(0, 185, 80, 0.9), rgba(0, 185, 80, 0.32) 30%, rgba(0, 185, 80, 0)),',
		'        linear-gradient(to top, rgba(0, 185, 80, 0.9), rgba(0, 185, 80, 0.32) 30%, rgba(0, 185, 80, 0)),',
		'        linear-gradient(to right, rgba(0, 185, 80, 0.85), rgba(0, 185, 80, 0.3) 30%, rgba(0, 185, 80, 0)),',
		'        linear-gradient(to left, rgba(0, 185, 80, 0.85), rgba(0, 185, 80, 0.3) 30%, rgba(0, 185, 80, 0)),',
		'        radial-gradient(circle at left top, rgba(0, 130, 60, 0.5), rgba(0, 130, 60, 0) 28vmin),',
		'        radial-gradient(circle at right top, rgba(0, 130, 60, 0.5), rgba(0, 130, 60, 0) 28vmin),',
		'        radial-gradient(circle at left bottom, rgba(0, 130, 60, 0.5), rgba(0, 130, 60, 0) 28vmin),',
		'        radial-gradient(circle at right bottom, rgba(0, 130, 60, 0.5), rgba(0, 130, 60, 0) 28vmin);',
		'    background-size: 100% 13vmin, 100% 13vmin, 16vmin 100%, 16vmin 100%, auto, auto, auto, auto;',
		'    background-position: left top, left bottom, left top, right top, left top, right top, left bottom, right bottom;',
		'    box-shadow: inset 0 0 3vmin 0 rgba(0, 60, 30, 0.18);',
		'}',
		'#copilotRelayGlow.relay-glow-bad {',
		'    background-image:',
		'        linear-gradient(to bottom, rgba(var(--relay-ce), 1), rgba(var(--relay-cm), 0.55) 30%, rgba(var(--relay-ct), 0)),',
		'        linear-gradient(to top, rgba(var(--relay-ce), 1), rgba(var(--relay-cm), 0.55) 30%, rgba(var(--relay-ct), 0)),',
		'        linear-gradient(to right, rgba(var(--relay-ce), 0.95), rgba(var(--relay-cm), 0.5) 30%, rgba(var(--relay-ct), 0)),',
		'        linear-gradient(to left, rgba(var(--relay-ce), 0.95), rgba(var(--relay-cm), 0.5) 30%, rgba(var(--relay-ct), 0)),',
		'        radial-gradient(circle at left top, rgba(var(--relay-ck), 0.72), rgba(var(--relay-ck), 0) var(--relay-cc)),',
		'        radial-gradient(circle at right top, rgba(var(--relay-ck), 0.72), rgba(var(--relay-ck), 0) var(--relay-cc)),',
		'        radial-gradient(circle at left bottom, rgba(var(--relay-ck), 0.72), rgba(var(--relay-ck), 0) var(--relay-cc)),',
		'        radial-gradient(circle at right bottom, rgba(var(--relay-ck), 0.72), rgba(var(--relay-ck), 0) var(--relay-cc));',
		'    background-size: 100% var(--relay-ey), 100% var(--relay-ey), var(--relay-ex) 100%, var(--relay-ex) 100%, auto, auto, auto, auto;',
		'    background-position: left top, left bottom, left top, right top, left top, right top, left bottom, right bottom;',
		'    box-shadow: inset 0 0 2vmin 0 rgba(220, 0, 0, 0.45);',
		'}',
		'#copilotRelayGlow.relay-glow-on {',
		'    opacity: 0.85;',
		'    animation: copilotRelayBreath 2.4s ease-in-out infinite alternate;',
		'}',
		'#copilotRelayFlash.relay-flash-ok {',
		'    background-image:',
		'        linear-gradient(to bottom, rgba(90, 255, 140, 0.95), rgba(90, 255, 140, 0.4) 32%, rgba(90, 255, 140, 0)),',
		'        linear-gradient(to top, rgba(90, 255, 140, 0.95), rgba(90, 255, 140, 0.4) 32%, rgba(90, 255, 140, 0)),',
		'        linear-gradient(to right, rgba(90, 255, 140, 0.9), rgba(90, 255, 140, 0.36) 32%, rgba(90, 255, 140, 0)),',
		'        linear-gradient(to left, rgba(90, 255, 140, 0.9), rgba(90, 255, 140, 0.36) 32%, rgba(90, 255, 140, 0));',
		'    background-size: 100% 16vmin, 100% 16vmin, 20vmin 100%, 20vmin 100%;',
		'    background-position: left top, left bottom, left top, right top;',
		'    box-shadow: inset 0 0 7vmin 1vmin rgba(90, 255, 140, 0.75);',
		'}',
		'#copilotRelayFlash.relay-flash-bad {',
		'    background-image:',
		'        linear-gradient(to bottom, rgba(255, 20, 20, 1), rgba(255, 20, 20, 0.5) 32%, rgba(255, 20, 20, 0)),',
		'        linear-gradient(to top, rgba(255, 20, 20, 1), rgba(255, 20, 20, 0.5) 32%, rgba(255, 20, 20, 0)),',
		'        linear-gradient(to right, rgba(255, 20, 20, 0.95), rgba(255, 20, 20, 0.46) 32%, rgba(255, 20, 20, 0)),',
		'        linear-gradient(to left, rgba(255, 20, 20, 0.95), rgba(255, 20, 20, 0.46) 32%, rgba(255, 20, 20, 0));',
		'    background-size: 100% 16vmin, 100% 16vmin, 20vmin 100%, 20vmin 100%;',
		'    background-position: left top, left bottom, left top, right top;',
		'    box-shadow: inset 0 0 7vmin 1vmin rgba(255, 20, 20, 0.8);',
		'}',
		'#copilotRelayFlash.relay-flash-run { animation: copilotRelayFlashOnce 420ms ease-out; }',
		'#copilotRelayFlash.relay-flash-bad.relay-flash-run { animation: copilotRelayFlashTwice 650ms ease-out; }',
		'@keyframes copilotRelayBreath { from { opacity: 0.42; } to { opacity: 0.95; } }',
		'@keyframes copilotRelayFlashOnce {',
		'    0% { opacity: 0; }',
		'    14% { opacity: 1; }',
		'    100% { opacity: 0; }',
		'}',
		'@keyframes copilotRelayFlashTwice {',
		'    0% { opacity: 0; }',
		'    12% { opacity: 1; }',
		'    42% { opacity: 0.2; }',
		'    58% { opacity: 0.95; }',
		'    100% { opacity: 0; }',
		'}'
	].join('\n');
	$('<style id="copilotRelayStyle"/>').text(relayStyle).appendTo('head');

	/* Live CFOP stage bar, meant for demoing a solve to non-cubers. */
	var stageBar = $('<div id="copilotStageBar"/>');
	var stageLabel = $('<span class="cp-stage-label"/>');
	var stageTip = $('<span id="copilotStageTip" class="cp-stage-tip"/>');
	var stageChips = [];
	var stagePlanKey = '';
	var stageArmed = false;
	var stageFacelet = null;
	var stageIdx = 0;
	var stagePopTid = 0;
	var stageHideTid = 0;
	var techToast = $('<div id="copilotTechToast"/>');
	var techToastLabel = $('<div class="cp-toast-label"/>');
	var techToastDesc = $('<div class="cp-toast-desc"/>');
	techToast.append($('<div class="cp-toast-star">★</div>'), techToastLabel, techToastDesc);
	var techTracker = cubeCopilot.makeTechniqueTracker();
	var techAch = [];
	var techToastKey = '';
	var techToastTid = 0;
	var techSeedFacelet = null;
	var techSeedProg = -1;
	var techF2LFacelet = null;
	var techPllFacelet = null;
	var TECH_LABELS = {
		xc: 'XC', xxc: 'XXC', xxxc: 'XXXC', xxxxc: 'XXXXC',
		zbls: 'ZBLS', wv: 'WV/SV', ols: 'OLS', ollskip: 'OLL Skip',
		zbll: 'ZBLL', coll: 'COLL', pll: 'PLL', epll: 'EPLL',
		ollcp: 'OLLCP', llll: '1LLL', llskip: 'LL Skip',
		multi: 'Multislotting'
	};
	var TECH_COLORS = {
		xc: 'green', xxc: 'green', xxxc: 'green', xxxxc: 'green',
		zbls: 'green', wv: 'green', ols: 'green', ollskip: 'green', multi: 'green',
		zbll: 'gold', coll: 'gold',
		pll: 'blue', epll: 'blue', ollcp: 'blue', llll: 'blue', llskip: 'gold'
	};
	var TECH_TEXT = {
		xc: ['COPILOT_TIP_XC', 'XC · cross + 1 F2L pair at once'],
		xxc: ['COPILOT_TIP_XXC', 'XXC · cross + 2 F2L pairs at once'],
		xxxc: ['COPILOT_TIP_XXXC', 'XXXC · cross + 3 F2L pairs at once'],
		xxxxc: ['COPILOT_TIP_XXXXC', 'XXXXC · cross + all 4 pairs at once'],
		zbls: ['COPILOT_TIP_ZBLS', 'ZBLS · last F2L pair + top cross at once'],
		wv: ['COPILOT_TIP_WV', 'WV/SV · last F2L pair + top corners oriented'],
		ols: ['COPILOT_TIP_OLS', 'OLS · last F2L pair + top layer oriented'],
		ollskip: ['COPILOT_TIP_OLLSKIP', 'OLL skip · top layer already oriented'],
		zbll: ['COPILOT_TIP_ZBLL', 'ZBLL%s · last layer in one algorithm'],
		coll: ['COPILOT_TIP_COLL', 'COLL%s · top corners in one algorithm'],
		pll: ['COPILOT_TIP_PLL', 'PLL%s · top layer in one algorithm'],
		epll: ['COPILOT_TIP_EPLL', 'EPLL%s · top edges in one algorithm'],
		ollcp: ['COPILOT_TIP_OLLCP', 'OLLCP · OLL + corner permutation at once'],
		llll: ['COPILOT_TIP_1LLL', '1LLL · last layer in one algorithm'],
		llskip: ['COPILOT_TIP_LLSKIP', 'LL skip · last layer already solved'],
		multi: ['COPILOT_TIP_MULTI', 'Multislotting · two F2L pairs at once']
	};
	var STAGE_PLANS = {
		cf4op: {label: 'CFOP', names: ['Cross', 'F1', 'F2', 'F3', 'F4', 'OLL', 'PLL'], base: 7},
		cfop: {label: 'CFOP', names: ['Cross', 'F2L', 'OLL', 'PLL'], base: 4},
		cf4o2p2: {label: 'CFOP', names: ['Cross', 'F1', 'F2', 'F3', 'F4', 'EOLL', 'OLL', 'CPLL', 'PLL'], base: 9},
		roux: {label: 'ROUX', names: ['FB', 'SB', 'CMLL', 'L6E'], base: 4}
	};
	var stageStyle = [
		'#copilotStageBar {',
		'    position: fixed;',
		'    left: 50%;',
		'    top: 7rem;',
		'    z-index: 42;',
		'    display: flex;',
		'    flex-wrap: wrap;',
		'    justify-content: center;',
		'    align-items: center;',
		'    max-width: 94vw;',
		'    box-sizing: border-box;',
		'    padding: 0.45rem 0.7rem;',
		'    border-radius: 0.6rem;',
		'    background: rgba(20, 20, 20, 0.55);',
		'    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);',
		'    font-family: Arial, sans-serif;',
		'    font-size: 1rem;',
		'    color: #fff;',
		'    pointer-events: none;',
		'    opacity: 0;',
		'    transform: translate(-50%, -8px);',
		'    transition: opacity 0.18s ease, transform 0.18s ease;',
		'}',
		'html.m #copilotStageBar { top: 8rem; font-size: 0.8rem; padding: 0.35rem 0.45rem; }',
		'#copilotStageBar.cp-stage-on { opacity: 1; transform: translate(-50%, 0); }',
		'.cp-stage-label {',
		'    margin-right: 0.35rem;',
		'    font-weight: bold;',
		'    letter-spacing: 0.06em;',
		'    color: #ffd54a;',
		'}',
		'.cp-stage-chip {',
		'    margin: 0 0.15rem;',
		'    min-width: 2.2em;',
		'    padding: 0.15rem 0.45rem;',
		'    border-radius: 0.3rem;',
		'    background: rgba(255, 255, 255, 0.14);',
		'    color: rgba(255, 255, 255, 0.5);',
		'    text-align: center;',
		'    white-space: nowrap;',
		'    transition: background 0.15s, color 0.15s, transform 0.15s;',
		'}',
		'html.m .cp-stage-chip { min-width: 1.6em; margin: 0 0.08rem; padding: 0.1rem 0.2rem; }',
		'.cp-stage-chip.cp-done { background: rgba(0, 185, 80, 0.9); color: #fff; }',
		'.cp-stage-chip.cp-cur {',
		'    background: #ffd54a;',
		'    color: #222;',
		'    font-weight: bold;',
		'    transform: scale(1.12);',
		'    box-shadow: 0 0 14px rgba(255, 213, 74, 0.85);',
		'}',
		'.cp-stage-chip.cp-pop { animation: copilotStagePop 520ms ease-out; }',
		'@keyframes copilotStagePop {',
		'    0% { transform: scale(1); box-shadow: none; }',
		'    40% { transform: scale(1.3); box-shadow: 0 0 16px rgba(90, 255, 140, 0.95); }',
		'    100% { transform: scale(1); box-shadow: none; }',
		'}',
		'.cp-stage-tip {',
		'    flex-basis: 100%;',
		'    margin-top: 0.15rem;',
		'    text-align: center;',
		'    font-size: 1em;',
		'    font-weight: bold;',
		'    line-height: 1.25;',
		'    color: #7ef29b;',
		'    white-space: nowrap;',
		'    overflow: hidden;',
		'    text-overflow: ellipsis;',
		'    display: none;',
		'}',
		'.cp-stage-tip.cp-tip-on { display: block; }',
		'.cp-stage-tip.cp-tip-combo {',
		'    color: rgba(255, 255, 255, 0.7);',
		'    font-size: 0.85em;',
		'    font-weight: normal;',
		'}',
		'.cp-stage-tip.cp-tip-green { color: #7ef29b; }',
		'.cp-stage-tip.cp-tip-gold { color: #ffd54a; }',
		'.cp-stage-tip.cp-tip-blue { color: #8fc8ff; }',
		'html.m .cp-stage-tip { font-size: 1.05em; }',
		'.cp-stage-tip.cp-tip-new { animation: copilotTipIn 280ms ease-out; }',
		'@keyframes copilotTipIn {',
		'    from { opacity: 0; transform: translateY(-4px); }',
		'    to { opacity: 1; transform: none; }',
		'}',
		'#copilotTechToast {',
		'    position: fixed;',
		'    left: 50%;',
		'    top: 12rem;',
		'    z-index: 44;',
		'    display: flex;',
		'    flex-direction: column;',
		'    align-items: center;',
		'    padding: 0.6rem 1.15rem 0.7rem;',
		'    border-radius: 0.8rem;',
		'    background: rgba(15, 15, 15, 0.86);',
		'    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.4), inset 0 0 0 2px var(--cp-tip-accent), 0 0 26px var(--cp-tip-glow);',
		'    font-family: Arial, sans-serif;',
		'    color: #fff;',
		'    pointer-events: none;',
		'    opacity: 0;',
		'    transform: translate(-50%, 0);',
		'}',
		'.cp-toast-star {',
		'    font-size: 1.1rem;',
		'    line-height: 1;',
		'    color: var(--cp-tip-accent);',
		'    margin-bottom: 0.2rem;',
		'}',
		'.cp-toast-label {',
		'    font-size: 1.55rem;',
		'    font-weight: bold;',
		'    letter-spacing: 0.04em;',
		'    color: var(--cp-tip-accent);',
		'    white-space: nowrap;',
		'}',
		'.cp-toast-desc {',
		'    margin-top: 0.1rem;',
		'    font-size: 0.9rem;',
		'    color: rgba(255, 255, 255, 0.88);',
		'    white-space: nowrap;',
		'}',
		'html.m .cp-toast-label { font-size: 1.3rem; }',
		'html.m .cp-toast-desc { font-size: 0.8rem; }',
		'#copilotTechToast.cp-toast-in { animation: copilotToastIn 420ms cubic-bezier(.2, 1.5, .4, 1) forwards; }',
		'#copilotTechToast.cp-toast-out { animation: copilotToastOut 440ms ease-in forwards; }',
		'@keyframes copilotToastIn {',
		'    0% { opacity: 0; transform: translate(-50%, 14px) scale(0.82); }',
		'    60% { opacity: 1; transform: translate(-50%, -3px) scale(1.05); }',
		'    100% { opacity: 1; transform: translate(-50%, 0) scale(1); }',
		'}',
		'@keyframes copilotToastOut {',
		'    0% { opacity: 1; transform: translate(-50%, 0) scale(1); }',
		'    100% { opacity: 0; transform: translate(-50%, -22px) scale(0.96); }',
		'}'
	].join('\n');
	$('<style id="copilotStageStyle"/>').text(stageStyle).appendTo('head');

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

	// Copilot hint methods.  Timing is independent: a solve may be timed with
	// any vrcMP, or not multi-phase at all, and still use the CFOP hints.
	var HINT_METHODS = {
		cfop: 1,
		lbl: 1,
		roux: 1
	};

	function hintMethod() {
		return kernel.getProp('cpMethod', 'cfop');
	}

	function methodEnabled() {
		return !!HINT_METHODS[hintMethod()];
	}

	function stageOf(facelet) {
		try {
			return cubeCopilot.getStage(facelet, hintMethod());
		} catch (e) {
			return null;
		}
	}

	function timeoutMs() {
		return Math.min(120, Math.max(0, ~~kernel.getProp('cpTimeout', 6))) * 1000;
	}


	function llMode() {
		return kernel.getProp('cpLlmode', 'op');
	}

	function highlightColor() {
		var val = String(kernel.getProp('cpColor', '#ffd700'));
		var m = /^#([0-9a-fA-F]{6})$/.exec(val);
		return m ? parseInt(m[1], 16) : 0xffd700;
	}

	/** how many off-track moves before the hint is re-planned */
	function maxUndo() {
		return Math.min(10, Math.max(1, ~~kernel.getProp('cpUndoMax', 3)));
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
		if (!isEnabled() || !isCube3() || !methodEnabled()) {
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
			timeLimit: req.timeLimit,
			llMode: req.llMode,
			method: req.method
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
		var stage = null;
		try {
			stage = stageOf(facelet);
		} catch (e) {}
		// the F2L hint is only a highlight: no search, no worker round-trip
		if (stage == 'f2l') {
			showF2lHint(facelet);
			return;
		}
		// Roux only has a CMLL hint; blocks/LSE have no algorithm to show
		if (hintMethod() == 'roux' && stage != 'cmll' && stage != 'solved') {
			return;
		}
		initWorker();
		postRequest({
			id: ++reqId,
			facelet: facelet,
			raw: curRawFacelet,
			manual: !!manual,
			timeLimit: manual ? 4000 : 1500,
			llMode: llMode(),
			method: hintMethod()
		});
	}

	function computeHintInline(facelet, manual) {
		try {
			cubeCopilot.warmUp();
			return cubeCopilot.getHint(facelet, {
				timeLimit: manual ? 4000 : 1500,
				llMode: llMode(),
				method: hintMethod()
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
		onTrackPairs = 0;
		f2lPair = null;
		lblPiece = null;
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
			stage = stageOf(facelet);
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

		var stageNow = null;
		try {
			stageNow = stageOf(facelet);
		} catch (e) {}
		// F2L: keep the live highlight on the corner/edge with the fewest moves
		// (an LL/CMLL algorithm may pass through F2L states, so let the hint
		// tracking below follow it instead of dropping the hint)
		if (stageNow == 'f2l' && !(hintInfo && hintTracks && hintTracks.length)) {
			if (hintInfo && hintInfo.stage != 'f2l') {
				finishHint();
			}
			if (f2lPair) {
				refreshF2lHint(facelet);
			}
			prevRaw = raw || prevRaw;
			updateStage(facelet, false);
			checkTimeout();
			return;
		}
		clearF2lHint();
		if (hintInfo && hintInfo.stage == 'cmll' && stageNow != 'cmll' &&
			!(hintTracks && hintTracks.length)) {
			finishHint();
		}
		// CF finished while a cross hint was active: go straight to the LL hint
		if ((stageNow == 'll' || stageNow == 'cmll') && hintInfo && hintInfo.stage != stageNow) {
			finishHint();
			updateStage(facelet, true);
			requestNext(facelet);
			return;
		}

		// keep an active hint alive while the user follows it
		if (hintInfo && hintTracks && hintTracks.length) {
			// off the algorithm: show the inverse moves that lead back to it
			if (offTrack) {
				// a pair was completed by another route: the hint is stale
				if (pairProgress(facelet) > onTrackPairs) {
					finishHint();
					updateStage(facelet, true);
					requestNext(facelet);
					return;
				}
				var utok = inferFromPrev(raw);
				if (utok) {
					if (hintUndo.length && utok == hintUndo[hintUndo.length - 1]) {
						hintUndo.pop();
					} else {
						hintUndo.push(cubeCopilot.invertToken(utok));
					}
					prevRaw = raw;
					// too many wrong moves: forget the undos and re-plan
					if (hintUndo.length > maxUndo()) {
						finishHint();
						updateStage(facelet, true);
						requestNext(facelet);
						return;
					}
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
			var exact = false;
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
					if (idx == track.cursor) {
						exact = true;
					}
					track.cursor = idx + 1;
					if (track.cursor >= track.steps.length) {
						track.done = true;
						track.pair.done = true;
						track.pair.remain = [];
					} else {
						track.pair.remain = track.steps[idx].remain.slice(0);
					}
				}
				if (!track.done) {
					allDone = false;
				}
			}
			if (matched) {
				// a jump that completed a pair means the position moved on
				if (!exact && pairProgress(facelet) > onTrackPairs) {
					finishHint();
					updateStage(facelet, true);
					requestNext(facelet);
					return;
				}
				hintLastSig = sig;
				prevRaw = raw || prevRaw;
				onTrackPairs = pairProgress(facelet);
				if (allDone) {
					finishHint();
					updateStage(facelet, true);
					requestNext(facelet);
				} else {
					refreshLblHighlight(facelet);
					renderCurrent();
					updateStage(facelet, false);
				}
				return;
			}
			// a whole-cube rotation (x/y/z) is pushed as a new facelet but does
			// not change the position, so it must not count as a wrong move
			if (sig && sig == hintLastSig) {
				// rotating the virtual cube changes the frame of the remaining
				// algorithm, so re-ask for it in the new orientation
				if (!isSmart && raw && prevRaw && raw != prevRaw) {
					prevRaw = raw;
					finishHint();
					updateStage(facelet, true);
					requestNext(facelet);
					return;
				}
				prevRaw = raw || prevRaw;
				updateStage(facelet, false);
				return;
			}
			// the user left the algorithm: say which moves undo the deviation
			if (pairProgress(facelet) > onTrackPairs) {
				finishHint();
				updateStage(facelet, true);
				requestNext(facelet);
				return;
			}
			var bad = inferFromPrev(raw);
			if (bad) {
				offTrack = true;
				hintUndo = [cubeCopilot.invertToken(bad)];
				hintInfo.undo = hintUndo.slice(0);
				prevRaw = raw;
				refreshLblHighlight(facelet);
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

		var prevStage = curStage;
		prevRaw = raw || prevRaw;
		updateStage(facelet, false);
		// reaching the last layer is a clear milestone: hint right away
		if ((curStage == 'll' || curStage == 'cmll') && prevStage != curStage) {
			requestNext(facelet);
			return;
		}
		// LBL: the beginner phases are short, so hint each new phase right away
		if (hintMethod() == 'lbl' && prevStage != curStage) {
			requestNext(facelet);
			return;
		}
		checkTimeout();
	}

	/** number of F2L pairs solved in this position */
	function pairProgress(facelet) {
		if (!facelet) {
			return 0;
		}
		try {
			return cubeCopilot.pairCount(facelet);
		} catch (e) {
			return 0;
		}
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
			stage = stageOf(facelet);
		} catch (e) {
			return;
		}
		if (stage != 'cross' && stage != 'f2l' && stage != 'll' && stage != 'cmll' &&
			stage != 'fl' && stage != 'sl' && stage != 'eoll' && stage != 'ocll' &&
			stage != 'cpll' && stage != 'epll') {
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
		if (curStage != 'f2l' && curStage != 'll' && curStage != 'cmll' &&
			curStage != 'fl' && curStage != 'sl' && curStage != 'eoll' &&
			curStage != 'ocll' && curStage != 'cpll' && curStage != 'epll') {
			return;
		}
		// never replace a hint the user is still following
		if (hintInfo || f2lPair) {
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
		// A relay attempt starts neutral (green) and keeps whatever pace colour
		// it has while solving. The final colour stays after the last cube, but
		// an aborted attempt (ESC) must not leave anything behind.
		if (status == -3 || status == -2) {
			relayFinished = false;
			if (relayActive()) {
				showRelayGlow(true);
			} else {
				clearRelayFeedback();
			}
		} else if (status == -1) {
			// idle: no breathing background until the next attempt starts
			if (!relayFinished) {
				clearRelayFeedback();
			}
		}
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
		onStageStatus(status);
	}

	// ------------------------------------------------------ relay feedback

	function relayTargetMs() {
		var val = parseFloat(kernel.getProp('cpRelayTarget', 0));
		return isFinite(val) && val > 0 ? val * 1000 : 0;
	}

	function relayActive() {
		return relayTargetMs() > 0 && /^r\d+$/.test(kernel.getProp('scrType'));
	}

	function clearRelayFeedback() {
		cancelRelayClear();
		setRelayDepth(0);
		relayGlow.removeClass('relay-glow-ok relay-glow-bad relay-glow-on');
		relayFlash.removeClass('relay-flash-ok relay-flash-bad relay-flash-run');
	}

	function cancelRelayClear() {
		if (relayClearTid) {
			clearTimeout(relayClearTid);
			relayClearTid = 0;
		}
	}

	/**
	 * The finishing colour stays just long enough to be read, then the overlay
	 * goes away on its own instead of lingering until the next attempt.
	 */
	function scheduleRelayClear(delay) {
		cancelRelayClear();
		relayClearTid = setTimeout(function() {
			relayClearTid = 0;
			clearRelayFeedback();
		}, delay);
	}

	function showRelayGlow(ok) {
		cancelRelayClear();
		relayGlow
			.removeClass('relay-glow-ok relay-glow-bad')
			.addClass(ok ? 'relay-glow-ok' : 'relay-glow-bad')
			.addClass('relay-glow-on');
	}

	/**
	 * How far behind the cumulative pace is (0 = on target, 1 = 60% over).
	 * The red gets darker and thicker as the gap grows.
	 */
	function setRelayDepth(depth) {
		var el = relayGlow[0];
		if (!el) {
			return;
		}
		var keys = ['--relay-ce', '--relay-cm', '--relay-ct', '--relay-ck', '--relay-ey', '--relay-ex', '--relay-cc'];
		var i;
		if (!(depth > 0)) {
			for (i = 0; i < keys.length; i++) {
				el.style.removeProperty(keys[i]);
			}
			return;
		}
		function mix(from, to) {
			return Math.round(from + (to - from) * depth);
		}
		el.style.setProperty('--relay-ce', mix(205, 145) + ', 0, 0');
		el.style.setProperty('--relay-cm', mix(165, 100) + ', 0, 0');
		el.style.setProperty('--relay-ct', mix(130, 70) + ', 0, 0');
		el.style.setProperty('--relay-ck', mix(110, 55) + ', 0, 0');
		el.style.setProperty('--relay-ey', (17 + 5 * depth).toFixed(2) + 'vmin');
		el.style.setProperty('--relay-ex', (20 + 5 * depth).toFixed(2) + 'vmin');
		el.style.setProperty('--relay-cc', (32 + 6 * depth).toFixed(2) + 'vmin');
	}

	/**
	 * Start this cube with a small flash, then breathe on the cumulative
	 * (x * target) edge colour until the next cube finishes.
	 */
	function setRelayFeedback(splitOk, cumulativeOk, depth) {
		showRelayGlow(cumulativeOk);
		setRelayDepth(cumulativeOk ? 0 : depth);

		relayFlash.removeClass('relay-flash-run relay-flash-ok relay-flash-bad');
		relayFlash.addClass(splitOk ? 'relay-flash-ok' : 'relay-flash-bad');
		void relayFlash[0].offsetHeight;
		relayFlash.addClass('relay-flash-run');
	}

	function onRelaySolve(value) {
		var target = relayTargetMs();
		if (!target) {
			relayFinished = false;
			clearRelayFeedback();
			return;
		}
		var idx = ~~value[0];
		var elapsed = ~~value[2];
		var split = ~~value[3];
		if (idx <= 0) {
			return;
		}
		var cumulativeOk = elapsed < idx * target;
		var splitOk = split < target;
		var paceBudget = idx * target;
		var depth = cumulativeOk ? 0 : Math.max(0, Math.min(1, (elapsed - paceBudget) / (paceBudget * 0.6)));
		relayFinished = !!value[4];
		setRelayFeedback(splitOk, cumulativeOk, depth);
		if (relayFinished) {
			scheduleRelayClear(1200);
		}
	}

	// ------------------------------------------------------ solve stage bar

	function stagePlan() {
		if (!/^(333|r\d+)$/.test(kernel.getProp('scrType', '333'))) {
			return '';
		}
		var mp = kernel.getProp('vrcMP', 'n');
		return STAGE_PLANS[mp] ? mp : '';
	}

	function buildStageChips(plan) {
		if (plan == stagePlanKey) {
			return false;
		}
		stagePlanKey = plan;
		stageIdx = 0;
		var def = STAGE_PLANS[plan];
		stageBar.empty().append(stageLabel.text(def.label));
		stageChips = [];
		for (var i = 0; i < def.names.length; i++) {
			stageChips.push($('<span class="cp-stage-chip"/>').text(def.names[i]).appendTo(stageBar));
		}
		stageBar.append(stageTip);
		return true;
	}

	function clearStageChips() {
		stagePlanKey = '';
		stageChips = [];
		stageIdx = 0;
		stageFacelet = null;
		stageBar.empty();
	}

	function popStageChip(chip) {
		chip.removeClass('cp-pop');
		void chip[0].offsetWidth;
		chip.addClass('cp-pop');
		if (stagePopTid) {
			clearTimeout(stagePopTid);
		}
		stagePopTid = setTimeout(function() {
			stagePopTid = 0;
			chip.removeClass('cp-pop');
		}, 560);
	}

	function paintStage(idx) {
		var n = stageChips.length;
		var def = STAGE_PLANS[stagePlanKey];
		var prev = stageIdx;
		stageIdx = idx;
		for (var i = 0; i < n; i++) {
			stageChips[i]
				.toggleClass('cp-done', i < idx)
				.toggleClass('cp-cur', i == idx);
		}
		stageLabel.text(def && idx >= n ? 'SOLVED' : (def ? def.label : ''));
		if (idx > prev && idx > 0 && idx <= n) {
			popStageChip(stageChips[idx - 1]);
		}
	}

	function stageIndexFor(plan, facelet) {
		var base = STAGE_PLANS[plan].base;
		var progress = cubeutil.getProgress(facelet, plan);
		if (typeof progress != 'number' || !isFinite(progress)) {
			return -1;
		}
		return Math.max(0, Math.min(base, base - progress));
	}

	function paintStageProgress(facelet, plan) {
		var idx = stageIndexFor(plan, facelet);
		if (idx < 0) {
			return false;
		}
		paintStage(idx);
		return true;
	}

	// --------------------------------------------------- technique tips

	function techLine(ach) {
		var txt = TECH_TEXT[ach.key];
		if (!txt) {
			return ach.label;
		}
		var str = L(txt[0], txt[1]);
		if (str.indexOf('%s') >= 0) {
			str = str.replace('%s', ach.name ? ' ' + ach.name : '');
		}
		return str;
	}

	function comboLine() {
		var parts = [];
		for (var i = 0; i < techAch.length; i++) {
			parts.push(techAch[i].label + (techAch[i].name ? ' ' + techAch[i].name : ''));
		}
		return parts.join(' → ');
	}

	function techParts(ach) {
		var full = techLine(ach) || ach.label || '';
		var sep = full.indexOf('·');
		if (sep > 0) {
			return {
				label: full.slice(0, sep).trim(),
				desc: full.slice(sep + 1).trim()
			};
		}
		return {
			label: ach.label + (ach.name ? ' ' + ach.name : ''),
			desc: full
		};
	}

	function renderTechCombo() {
		var txt = comboLine();
		stageTip.removeClass('cp-tip-green cp-tip-gold cp-tip-blue');
		if (!txt) {
			stageTip.removeClass('cp-tip-on cp-tip-combo').text('');
			return;
		}
		stageTip.text(txt).addClass('cp-tip-on cp-tip-combo');
	}

	function techAccent(cat) {
		return cat == 'gold' ? '#ffd54a' : (cat == 'blue' ? '#8fc8ff' : '#7ef29b');
	}

	function techGlow(cat) {
		return cat == 'gold' ? 'rgba(255, 213, 74, 0.35)' :
			(cat == 'blue' ? 'rgba(143, 200, 255, 0.3)' : 'rgba(126, 242, 155, 0.3)');
	}

	function positionTechToast() {
		var rect = stageBar.length ? stageBar[0].getBoundingClientRect() : null;
		var barBottom = (rect && rect.height) ? rect.bottom : $(window).height() * 0.18;
		var top = barBottom + 10;
		var left = '50%';
		var width = techToast[0] ? techToast[0].offsetWidth : 180;
		var height = techToast[0] ? techToast[0].offsetHeight : 80;
		var cube = $('#container').find('canvas, .touchcube').first();
		var cr = cube.length ? cube[0].getBoundingClientRect() : null;
		var stats = $('#stats')[0];
		if (cr && cr.width > 0 && cr.height > 0) {
			// On wide screens there is usually a free column between the stats
			// panel and the cube; use it instead of covering the cube.
			var gapLeft = (stats ? stats.getBoundingClientRect().right : 0) + 8;
			var gapRight = cr.left - 8;
			if (gapRight - gapLeft >= width + 8) {
				left = Math.round((gapLeft + gapRight) / 2) + 'px';
				top = Math.max(barBottom + 10, cr.top + cr.height * 0.06);
			} else if (cr.top - (barBottom + 10) < height + 8 &&
					$(window).height() - cr.bottom >= height + 70) {
				top = cr.bottom + 12;
			}
		}
		top = Math.max(60, Math.min(top, $(window).height() * 0.45));
		techToast.css({
			'left': left,
			'top': Math.round(top) + 'px'
		});
	}

	function showTechToast(ach) {
		var parts = techParts(ach);
		techToastKey = ach.key;
		techToastLabel.text(parts.label);
		techToastDesc.text(parts.desc);
		positionTechToast();
		techToast[0].style.setProperty('--cp-tip-accent', techAccent(ach.cat));
		techToast[0].style.setProperty('--cp-tip-glow', techGlow(ach.cat));
		techToast.removeClass('cp-toast-out cp-toast-in');
		void techToast[0].offsetWidth;
		techToast.addClass('cp-toast-in');
		if (techToastTid) {
			clearTimeout(techToastTid);
			techToastTid = 0;
		}
		techToastTid = setTimeout(function() {
			techToast.removeClass('cp-toast-in').addClass('cp-toast-out');
			techToastTid = setTimeout(function() {
				techToastTid = 0;
				techToastKey = '';
				techToast.removeClass('cp-toast-out');
				techToastLabel.text('');
				techToastDesc.text('');
			}, 460);
		}, 1700);
	}

	function hideTechToast() {
		if (techToastTid) {
			clearTimeout(techToastTid);
			techToastTid = 0;
		}
		techToastKey = '';
		techToast.removeClass('cp-toast-in cp-toast-out');
		techToastLabel.text('');
		techToastDesc.text('');
	}

	function refreshTechText() {
		renderTechCombo();
		if (!techToastKey) {
			return;
		}
		for (var i = 0; i < techAch.length; i++) {
			if (techAch[i].key == techToastKey) {
				var parts = techParts(techAch[i]);
				techToastLabel.text(parts.label);
				techToastDesc.text(parts.desc);
				return;
			}
		}
	}

	function requestTechName(ach, facelet, mode, wantKind) {
		if (ach.req || ach.name || !facelet || !workerUsable()) {
			return;
		}
		ach.req = true;
		try {
			csTimerWorker.getHint(facelet, {
				method: 'cfop',
				llMode: mode,
				timeLimit: 2000
			}, function(hint) {
				if (!hint || hint.kind != wantKind || !hint.name) {
					return;
				}
				ach.name = hint.name;
				refreshTechText();
			});
		} catch (e) {}
	}

	function onTechEvent(ev) {
		var ach = null;
		for (var i = 0; i < techAch.length; i++) {
			if (techAch[i].key == ev.key) {
				ach = techAch[i];
				break;
			}
		}
		if (!ach) {
			ach = {
				key: ev.key,
				label: TECH_LABELS[ev.key] || ev.key,
				name: '',
				cat: TECH_COLORS[ev.key] || 'green',
				req: false
			};
			techAch.push(ach);
		}
		renderTechCombo();
		showTechToast(ach);
		if (ev.key == 'zbll' || ev.key == 'coll' || ev.key == 'pll') {
			requestTechName(ach, techF2LFacelet, ev.key == 'pll' ? 'op' : ev.key, ev.key);
		} else if (ev.key == 'epll') {
			var tst = techTracker.state();
			var facelet = (tst && tst.llEo && tst.llCo) ? techF2LFacelet : techPllFacelet;
			requestTechName(ach, facelet || techF2LFacelet, 'op', 'pll');
		}
	}

	function techStep(facelet, prog) {
		var sn = {prog: prog};
		if (prog <= 3) {
			var st = null;
			try {
				st = cubeCopilot.techniqueState(facelet);
			} catch (e) {}
			if (st) {
				sn.eo = st.eo;
				sn.co = st.co;
				sn.cp = st.cp;
				sn.solved = st.solved;
			}
		}
		if (prog <= 2 && !techF2LFacelet) {
			techF2LFacelet = facelet;
		}
		if (prog == 1 && sn.eo && sn.co && sn.cp && !techPllFacelet) {
			techPllFacelet = facelet;
		}
		var evs = techTracker.step(sn);
		for (var i = 0; i < evs.length; i++) {
			onTechEvent(evs[i]);
		}
	}

	function resetTech(keepToast, seedProg) {
		if (!keepToast) {
			hideTechToast();
		}
		techAch = [];
		techF2LFacelet = null;
		techPllFacelet = null;
		techSeedProg = (typeof seedProg == 'number') ? seedProg : -1;
		if (techSeedProg < 0 && techSeedFacelet) {
			try {
				techSeedProg = cubeutil.getProgress(techSeedFacelet, 'cf4op');
			} catch (e) {}
		}
		techTracker.reset(techSeedProg);
		stageTip.removeClass('cp-tip-on cp-tip-combo cp-tip-green cp-tip-gold cp-tip-blue cp-tip-new').text('');
	}

	// Sit just below the scramble, so the bar follows wrapped scramble text
	// instead of covering the timer or the bottom toolbar.
	function placeStageBar() {
		var txt = $('#scrambleTxt');
		var y = txt.length ? txt[0].getBoundingClientRect().bottom + 8 : 0;
		if (!(y > 0)) {
			y = 120;
		}
		stageBar.css('top', Math.min(y, $(window).height() * 0.45) + 'px');
	}

	function showStageBar() {
		if (!stageChips.length) {
			return;
		}
		if (stageHideTid) {
			clearTimeout(stageHideTid);
			stageHideTid = 0;
		}
		placeStageBar();
		stageBar.addClass('cp-stage-on');
	}

	function hideStageBar(delay) {
		if (stageHideTid) {
			clearTimeout(stageHideTid);
			stageHideTid = 0;
		}
		if (delay > 0) {
			stageHideTid = setTimeout(function() {
				stageHideTid = 0;
				stageBar.removeClass('cp-stage-on');
			}, delay);
		} else {
			stageBar.removeClass('cp-stage-on');
		}
	}

	function onStageStatus(status) {
		if (status >= 1) {
			// Multi-phase timers count down through status values >= 1 on
			// every completed phase; only the first one starts a new cube.
			if (stageArmed) {
				return;
			}
			stageArmed = true;
			stageFacelet = null;
			resetTech();
			var plan = stagePlan();
			if (plan) {
				buildStageChips(plan);
				paintStage(0);
				showStageBar();
			} else {
				hideStageBar(0);
			}
			return;
		}
		var wasArmed = stageArmed;
		stageArmed = false;
		if (!wasArmed) {
			return;
		}
		// keep a finished stepper visible for a moment, but never leave
		// anything behind when the attempt is aborted
		var solved = stageChips.length > 0 && stageIdx >= stageChips.length;
		hideStageBar(solved ? 2600 : 0);
		if (!solved) {
			hideTechToast();
		}
	}

	/**
	 * A relay moves to the next cube while the timer keeps running and the
	 * next scramble is applied behind the scenes, so no virtualState arrives
	 * until the first user move.  Reset here, or the stepper stays on the
	 * previous cube's SOLVED/PLL state.
	 */
	function restartStageForNextCube() {
		stageFacelet = null;
		techSeedFacelet = null;
		resetTech(true, 7);
		if (stageChips.length) {
			paintStage(0);
			showStageBar();
		}
	}

	function onStageFacelet(facelet) {
		if (!facelet || facelet.length != 54) {
			return;
		}
		if (!stageArmed) {
			// remember the scrambled state so the first move can be judged
			techSeedFacelet = facelet;
			return;
		}
		var plan = stagePlan();
		if (!plan) {
			return;
		}
		buildStageChips(plan);
		stageFacelet = facelet;
		var prog = cubeutil.getProgress(facelet, 'cf4op');
		if (plan != 'roux') {
			techStep(facelet, prog);
		}
		var idx = stageIndexFor(plan, facelet);
		if (idx < 0) {
			return;
		}
		// One alg often breaks a finished stage for a move or two (an F2L
		// insert turns cross edges out of the D layer).  The multi-phase
		// timer never rolls back, so the stepper must not either.
		if (idx > stageIdx) {
			paintStage(idx);
		}
		showStageBar();
	}

	function refreshStage() {
		var plan = stagePlan();
		if (!plan) {
			clearStageChips();
			stageBar.empty().removeClass('cp-stage-on');
			return;
		}
		buildStageChips(plan);
		if (stageArmed && stageFacelet) {
			paintStageProgress(stageFacelet, plan);
		} else {
			paintStage(0);
		}
		if (stageArmed) {
			showStageBar();
		}
	}

	// ---------------------------------------------------------------- display

	function hintCategory(hint) {
		if (hint.stage == 'f2l') {
			return L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', hint.name);
		}
		if (hint.kind == 'lblcorner') {
			return L('COPILOT_LBL_CORNER', 'First-layer corner %s').replace('%s', hint.name);
		}
		if (hint.kind == 'lblmid') {
			return L('COPILOT_LBL_MID', 'Second-layer edge %s').replace('%s', hint.name);
		}
		if (hint.stage == 'eoll') {
			return L('COPILOT_LBL_EOLL', 'Top layer: orient the edges (cross)');
		}
		if (hint.stage == 'ocll') {
			return L('COPILOT_LBL_OCLL', 'Top layer: orient the corners');
		}
		if (hint.stage == 'cpll') {
			return L('COPILOT_LBL_CPLL', 'Top layer: place the corners');
		}
		if (hint.stage == 'epll') {
			return L('COPILOT_LBL_EPLL', 'Top layer: place the edges');
		}
		if (hint.kind == 'oll') {
			return L('COPILOT_OLL', 'OLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'pll') {
			return L('COPILOT_PLL', 'PLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'coll') {
			return L('COPILOT_COLL', 'COLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'zbll') {
			return L('COPILOT_ZBLL', 'ZBLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'cmll') {
			if (hint.name == 'AUF') {
				return L('COPILOT_AUF', 'Adjust the top layer');
			}
			return L('COPILOT_CMLL', 'CMLL %s').replace('%s', hint.name);
		}
		if (hint.kind == 'auf') {
			return L('COPILOT_AUF', 'Adjust the top layer');
		}
		if (hint.kind == 'cross') {
			return L('COPILOT_CROSS', 'Finish the cross');
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
			if (hint.kind == 'cross') {
				return L('COPILOT_NO_CROSS', 'No cross hint found');
			}
			if (hint.kind == 'cmll') {
				return L('COPILOT_NO_CMLL', 'No algorithm for this CMLL case');
			}
			if (hint.stage == 'fl' || hint.stage == 'sl' || hint.stage == 'eoll' ||
				hint.stage == 'ocll' || hint.stage == 'cpll' || hint.stage == 'epll') {
				return L('COPILOT_NO_ALG', 'No algorithm for this case');
			}
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

	/** undo moves first, then the rest of the algorithm (one sequence) */
	function displayMoves(hint, pair) {
		var undo = (hint && hint.undo && hint.undo.length) ? hint.undo.slice().reverse() : [];
		return undo.concat(pair.remain || []);
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
						$('<span/>').css({'font-size': '1.15em'}).text(displayMoves(hint, active[i]).join(' '))
					)
				);
			}
			return;
		}
		var pair = active[0];
		var cat = hint.stage == 'f2l' ? L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', pair.name) : hintCategory(hint);
		target.append(
			$('<div/>').html(cat),
			$('<div/>').css({'font-size': '1.3em'}).text(displayMoves(hint, pair).join(' '))
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
		var total = hint.pairs ? hint.pairs.length : 1;
		if (hint.stage == 'f2l' && total > 1) {
			hintOverlay.append($('<div class="cp-cat"/>').html(L('COPILOT_F2L_PAIRS', 'Next F2L pairs:')));
			for (var i = 0; i < active.length; i++) {
				hintOverlay.append(
					$('<div class="cp-pair"/>').append(
						$('<span class="cp-name"/>').text(active[i].name + ': '),
						$('<span class="cp-alg2"/>').text(displayMoves(hint, active[i]).join(' '))
					)
				);
			}
		} else {
			hintOverlay.append(
				$('<div class="cp-cat"/>').html(hint.stage == 'f2l' ?
					L('COPILOT_F2L', 'Next F2L pair: %s').replace('%s', active[0].name) : hintCategory(hint)),
				$('<div class="cp-alg"/>').text(displayMoves(hint, active[0]).join(' '))
			);
		}
		hintOverlay.show();
	}

	function hideOverlay() {
		hintOverlay.hide().empty();
	}

	function highlightFacelets(slots) {
		try {
			if (timer.virtual && timer.virtual.highlight) {
				timer.virtual.highlight(slots && slots.length ? slots : null, highlightColor());
			}
		} catch (e) {}
	}

	/**
	 * F2L hint: only the corner and edge of the pair that needs the fewest
	 * moves are highlighted.  No algorithm is computed or shown.
	 */
	function showF2lHint(facelet) {
		var info = null;
		try {
			info = cubeCopilot.minPair(facelet);
		} catch (e) {}
		// drop any hint still in flight for an earlier position
		lastSentId = ++reqId;
		queuedReq = null;
		if (hintInfo) {
			finishHint();
		}
		if (!info) {
			f2lPair = null;
			clearHighlight();
			return;
		}
		f2lPair = info;
		hintedFacelet = facelet;
		lastWarnAt = $.now();
		stageSince = $.now();
		onTrackPairs = pairProgress(facelet);
		hintSpan.empty();
		if (toolHintSpan) {
			toolHintSpan.empty();
		}
		hideOverlay();
		highlightFacelets(info.pieces);
	}

	/** keep the highlight on the same corner/edge while they are being moved */
	function refreshF2lHint(facelet) {
		if (!f2lPair) {
			return;
		}
		var info = null;
		try {
			// colours rather than slots: turning the whole cube must not swap
			// the highlighted pair for whichever pair now occupies that slot
			info = cubeCopilot.locatePair(facelet, f2lPair.cornerColors, f2lPair.edgeColors);
		} catch (e) {}
		if (!info || info.solved) {
			f2lPair = null;
			clearHighlight();
			updateStage(facelet, true);
			var stage = null;
			try {
				stage = stageOf(facelet);
			} catch (e) {}
			if (stage == 'f2l') {
				showF2lHint(facelet);
			}
			return;
		}
		f2lPair.pieces = info.pieces;
		f2lPair.cornerAt = info.cornerAt;
		f2lPair.edgeAt = info.edgeAt;
		f2lPair.solved = false;
		highlightFacelets(info.pieces);
	}

	function clearF2lHint() {
		if (!f2lPair) {
			return;
		}
		f2lPair = null;
		clearHighlight();
	}

	/** keep the highlight on the corner/edge a first/second-layer hint targets */
	function refreshLblHighlight(facelet) {
		if (!lblPiece || !facelet) {
			return;
		}
		var info = null;
		try {
			info = cubeCopilot.locateLbl(facelet, lblPiece);
		} catch (e) {}
		if (!info) {
			return;
		}
		lblPiece.pieces = info.pieces;
		highlightFacelets(info.pieces);
	}

	function renderCurrent() {
		if (!hintInfo) {
			return;
		}
		renderHint(hintInfo, hintSpan);
		if (toolHintSpan) {
			renderHint(hintInfo, toolHintSpan);
		}
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
		// the position moved on while the hint was being computed
		if (facelet && curFacelet && facelet != curFacelet) {
			return;
		}
		f2lPair = null;
		baseHint = hint;
		hintInfo = composeHint(hint);
		// first/second-layer hints also highlight the piece they target
		if (hint.kind == 'lblcorner' || hint.kind == 'lblmid') {
			lblPiece = {
				kind: hint.kind,
				cornerColors: hint.cornerColors,
				edgeColors: hint.edgeColors,
				pieces: hint.pieces
			};
			highlightFacelets(hint.pieces);
		} else if (lblPiece) {
			lblPiece = null;
			clearHighlight();
		}
		hintFacelet = facelet;
		hintLastSig = null;
		hintUndo = [];
		offTrack = false;
		hintInfo.undo = [];
		prevRaw = raw || null;
		onTrackPairs = pairProgress(facelet);
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
		var status = L('COPILOT_DISABLED', 'Copilot is off');
		if (isEnabled()) {
			status = methodEnabled() ? L('COPILOT_WAITING', 'Waiting for a solve...') :
				L('COPILOT_METHOD_OFF_MSG', 'Hint method is off');
		}
		showStatus(status);
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
		onTrackPairs = pairProgress(hintFacelet);
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
			var gFacelet = smartFacelet(value[0]);
			onFacelet(gFacelet, value[0]);
			onStageFacelet(gFacelet);
		} else if (signal == 'virtualState') {
			isSmart = false;
			smartOri = null;
			onFacelet(value[0], value[0]);
			onStageFacelet(value[0]);
		} else if (signal == 'timerStatus') {
			onTimerStatus(value);
		} else if (signal == 'relaySolve') {
			onRelaySolve(value);
			if (stageArmed && !value[4] && /^r\d+$/.test(kernel.getProp('scrType'))) {
				restartStageForNextCube();
			}
		} else if (signal == 'relayReset') {
			relayFinished = false;
			clearRelayFeedback();
			hideTechToast();
		} else if (signal == 'property') {
			if (/^(cpEnabled|cpMethod|cpTimeout|cpLlmode|cpUndoMax)$/.exec(value[0])) {
				clearAll();
				refreshStatus();
				if (isEnabled() && methodEnabled()) {
					startPolling();
				} else {
					stopPolling();
				}
			} else if (value[0] == 'cpColor') {
				if (f2lPair && curFacelet) {
					highlightFacelets(f2lPair.pieces);
				}
			} else if (value[0] == 'cpCustom') {
				rebuildHintFromDefault();
			} else if (value[0] == 'cpRelayTarget') {
				relayFinished = false;
				clearRelayFeedback();
			} else if (value[0] == 'vrcMP' || value[0] == 'scrType') {
				refreshStage();
			}
		} else if (signal == 'scramble' || signal == 'time') {
			clearAll();
			if (signal == 'scramble') {
				techSeedFacelet = null;
				if (!relayActive()) {
					relayFinished = false;
					clearRelayFeedback();
				}
			}
		}
	}

	function initProps() {
		kernel.regProp('copilot', 'cpEnabled', 0, L('COPILOT_ENABLED', 'Enable Copilot'), [false], 1);
		kernel.regProp('copilot', 'cpMethod', 1, L('COPILOT_METHOD', 'Hint method'), ['cfop', ['cfop', 'lbl', 'roux', 'off'], [
			L('COPILOT_METHOD_CFOP', 'CFOP (Cross/F2L/LL)'),
			L('COPILOT_METHOD_LBL', 'Layer by layer (beginner)'),
			L('COPILOT_METHOD_ROUX', 'Roux (CMLL)'),
			L('COPILOT_METHOD_OFF', 'Off')
		]], 1);
		kernel.regProp('copilot', 'cpColor', 3, L('COPILOT_COLOR', 'Highlight color'), ['#ffd700'], 1);
		kernel.regProp('copilot', 'cpTimeout', 2, L('COPILOT_TIMEOUT', 'Hint after no progress (s)'), [6, 0, 120], 1);
		kernel.regProp('copilot', 'cpUndoMax', 2, L('COPILOT_UNDOMAX', 'Re-plan after this many wrong moves'), [3, 1, 10], 1);
		kernel.regProp('copilot', 'cpRelayTarget', ~2, L('COPILOT_RELAY_TARGET', '3x3 relay target per cube (s, 0 = off)'), [0, 0, 999], 1);
		kernel.regProp('copilot', 'cpLlmode', 1, L('COPILOT_LLMODE', 'Last-layer method'), ['op', ['op', 'coll', 'zbll', 'alll', 'off'], [
			L('COPILOT_LL_OP', 'OLL + PLL'),
			L('COPILOT_LL_COLL', 'COLL + EPLL'),
			L('COPILOT_LL_ZBLL', 'ZBLL'),
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
		kernel.regListener('copilot', 'relaySolve', procSignal);
		kernel.regListener('copilot', 'relayReset', procSignal);
		kernel.regListener('copilot', 'timerStatus', procSignal);
		kernel.regListener('copilot', 'property', procSignal, /^(cpEnabled|cpMethod|cpColor|cpTimeout|cpLlmode|cpUndoMax|cpCustom|cpRelayTarget|vrcMP|scrType)$/);
		kernel.regListener('copilot', 'scramble', procSignal);
		kernel.regListener('copilot', 'time', procSignal);
		$(window).on('resize', function() {
			if (stageBar.hasClass('cp-stage-on')) {
				placeStageBar();
			}
			if (techToastKey) {
				positionTechToast();
			}
		});

		initProps();

		kernel.addWindow('copilot', L('COPILOT_BUTTON', 'HINT'), div, false, false, 5);
		tools.regTool('copilot', L('TOOLS_COPILOT', 'Copilot'), toolExec);
		relayGlow.appendTo('body');
		relayFlash.appendTo('body');
		stageBar.appendTo('body');
		techToast.appendTo('body');
		hintOverlay.appendTo('body');
		chkEnable.prop('checked', isEnabled());
		refreshStatus();
		refreshStage();
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
