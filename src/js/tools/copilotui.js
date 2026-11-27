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
	var techTracker = cubeCopilot.makeTechniqueTracker();
	var techAch = [];
	var techToastItems = [];
	var tipAudio = null;
	var tipAudioGain = null;
	var tipSoundAt = 0;
	var techSeedFacelet = null;
	var techSeedProg = -1;
	var techF2LFacelet = null;
	var techPllFacelet = null;
	var techSteps = 0;
	var techFaceNo = 0;
	var techHist = [];
	var techPairMask = 0;
	var techFlow = 0;
	var techFlowCF = false;
	var techFlowFF = false;
	var techBasic = 0;
	var techMoveTs = [];
	var techTpsMax = 0;
	var techTpsTier = 0;
	var techTpsHit = [];
	var techSolveStart = 0;
	var techOpenMs = 0;
	var techLastGap = 0;
	var techMaxGap = 0;
	var techRotRun = 0;
	var techRotMax = 0;
	var techRotated = false;
	var techNorotDone = false;
	var techUseMoves = false;
	var techSmartCenter = null;
	var techMaxProg = 0;
	var techLastProg = -1;
	var lastSolveSolved = false;
	var achTracker = cubeCopilot.makeAchievementTracker();
	var achToast = $('<div id="copilotAchToast"/>');
	var achKicker = $('<div class="cp-ach-kicker"/>');
	var achLabel = $('<div class="cp-ach-label"/>');
	var achDesc = $('<div class="cp-ach-desc"/>');
	var achMore = $('<div class="cp-ach-more"/>');
	var achBadge = $('<div class="cp-ach-badge"/>');
	achToast.append(achKicker, achLabel, achDesc, achMore, achBadge);
	var achTid = 0;
	var achPanel = $('<div id="copilotAchPanel"/>');
	var achRows = {};
	var browseAchBtn = $('<span class="cp-ach-browse click"/>');
	var achSummary = $('<span class="cp-ach-summary"/>');
	var achBook = null;
	var achBookGrid = null;
	var achBookCount = null;
	var achBookTabs = null;
	var achBookFilter = 'all';
	var toolAchBtn = null;
	var toolAchSpan = null;
	var comboBadge = $('<span class="cp-stage-combo"/>');
	var TECH_LABELS = {
		xc: 'XC', xxc: 'XXC', xxxc: 'XXXC', xxxxc: 'XXXXC',
		zbls: 'ZBLS', wv: 'WV/SV', ols: 'OLS', ollskip: 'OLL Skip',
		zbll: 'ZBLL', coll: 'COLL', pll: 'PLL', epll: 'EPLL',
		ollcp: 'OLLCP', llll: '1LLL', llskip: 'LL Skip',
		multi: 'Multislotting', basicf2l: 'Basic F2L',
		flowcf: 'Cross→F1', flowff: 'F2F', flowfl: 'F→LL', flowop: 'OLL→PLL',
		tps6: 'TPS 6', tps8: 'TPS 8', tps10: 'TPS 10', norot: 'No Rot'
	};
	var TECH_COLORS = {
		xc: 'green', xxc: 'green', xxxc: 'green', xxxxc: 'green',
		zbls: 'green', wv: 'green', ols: 'green', ollskip: 'green', multi: 'green',
		basicf2l: 'green', flowcf: 'green', flowff: 'green', flowfl: 'green',
		tps6: 'green', tps8: 'gold', tps10: 'gold', norot: 'gold',
		zbll: 'gold', coll: 'gold',
		pll: 'blue', epll: 'blue', ollcp: 'blue', llll: 'blue', llskip: 'gold',
		flowop: 'blue'
	};
	var TECH_TEXT = {
		xc: ['COPILOT_TIP_XC', 'XC · cross + 1 F2L pair at once'],
		xxc: ['COPILOT_TIP_XXC', 'XXC · cross + 2 F2L pairs at once'],
		xxxc: ['COPILOT_TIP_XXXC', 'XXXC · cross + 3 F2L pairs at once'],
		xxxxc: ['COPILOT_TIP_XXXXC', 'XXXXC · cross + all 4 pairs at once'],
		zbls: ['COPILOT_TIP_ZBLS', 'ZBLS · last F2L pair + top cross at once'],
		wv: ['COPILOT_TIP_WV', 'WV/SV · last F2L pair + top corners oriented'],
		ols: ['COPILOT_TIP_OLS', 'OLS · last F2L pair + top layer oriented'],
		ollskip: ['COPILOT_TIP_OLLSKIP', 'OLL skip · top layer oriented when F2L ends'],
		zbll: ['COPILOT_TIP_ZBLL', 'ZBLL%s · last layer in one algorithm'],
		coll: ['COPILOT_TIP_COLL', 'COLL%s · top corners in one algorithm'],
		pll: ['COPILOT_TIP_PLL', 'PLL%s · top layer in one algorithm'],
		epll: ['COPILOT_TIP_EPLL', 'EPLL%s · top edges in one algorithm'],
		ollcp: ['COPILOT_TIP_OLLCP', 'OLLCP · OLL + corner permutation at once'],
		llll: ['COPILOT_TIP_1LLL', '1LLL · last layer in one algorithm'],
		llskip: ['COPILOT_TIP_LLSKIP', 'LL skip · last layer already solved'],
		multi: ['COPILOT_TIP_MULTI', 'Multislotting · two F2L pairs at once'],
		basicf2l: ['COPILOT_TIP_BASICF2L', 'Basic F2L · three move insert'],
		flowcf: ['COPILOT_TIP_FLOWCF', 'Cross→F1 · straight on without a pause'],
		flowff: ['COPILOT_TIP_FLOWFF', 'F2F · next pair without a pause'],
		flowfl: ['COPILOT_TIP_FLOWFL', 'F→LL · last F2L pair into OLL without a pause'],
		flowop: ['COPILOT_TIP_FLOWOP', 'OLL→PLL · OLL into PLL without a pause'],
		tps6: ['COPILOT_TIP_TPS6', 'TPS 6 · six turns per second'],
		tps8: ['COPILOT_TIP_TPS8', 'TPS 8 · eight turns per second'],
		tps10: ['COPILOT_TIP_TPS10', 'TPS 10 · ten turns per second'],
		norot: ['COPILOT_TIP_NOROT',
			'No Rotation · all 4 F2L pairs and the whole last layer without turning the cube']
	};
	var ACH_META = {
		hot3: ['COPILOT_ACH_HOT3', 'On Fire',
			'COPILOT_ACH_HOT3_D', 'Trigger a technique in 3 solves in a row', 'gold'],
		zb3: ['COPILOT_ACH_ZB3', 'ZB Maniac',
			'COPILOT_ACH_ZB3_D', '3 ZBLL solves in a row', 'gold'],
		coll3: ['COPILOT_ACH_COLL3', 'COLL Harvester',
			'COPILOT_ACH_COLL3_D', '3 COLL solves in a row', 'purple'],
		slot3: ['COPILOT_ACH_SLOT3', 'Slot Artist',
			'COPILOT_ACH_SLOT3_D', '3 OLS / ZBLS / WV solves in a row', 'purple'],
		xc3: ['COPILOT_ACH_XC3', 'Cross Crusader',
			'COPILOT_ACH_XC3_D', '3 XC-or-better solves in a row', 'blue'],
		flow3: ['COPILOT_ACH_FLOW3', 'Flow State',
			'COPILOT_ACH_FLOW3_D', '3 connected stage changes in one solve', 'purple'],
		basic3: ['COPILOT_ACH_BASIC3', 'Back to Basics',
			'COPILOT_ACH_BASIC3_D', '3 basic three-move F2L inserts in one solve', 'blue'],
		tps3: ['COPILOT_ACH_TPS3', 'Speed Demon',
			'COPILOT_ACH_TPS3_D', '3 solves in a row peaking at 8+ TPS', 'blue'],
		speed: ['COPILOT_ACH_SPEED', 'Lightspeed',
			'COPILOT_ACH_SPEED_D', 'Reach 10 TPS in one solve', 'purple'],
		norot: ['COPILOT_ACH_NOROT', 'Four-Way F2L',
			'COPILOT_ACH_NOROT_D',
			'All 4 F2L pairs and the whole last layer without a cube rotation', 'gold'],
		fullhouse: ['COPILOT_ACH_FULLHOUSE', 'Full House',
			'COPILOT_ACH_FULLHOUSE_D', 'ZBLS + ZBLL in one solve', 'gold'],
		triple: ['COPILOT_ACH_TRIPLE', 'Triple Combo',
			'COPILOT_ACH_TRIPLE_D', '3 techniques in one solve', 'purple'],
		codex: ['COPILOT_ACH_CODEX', 'Codex Master',
			'COPILOT_ACH_CODEX_D', 'Collect every technique', 'gold']
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
		'    gap: 0.45rem;',
		'    font-family: Arial, sans-serif;',
		'    color: #fff;',
		'    pointer-events: none;',
		'    transform: translateX(-50%);',
		'}',
		'#copilotTechToast .cp-toast {',
		'    display: flex;',
		'    flex-direction: column;',
		'    align-items: center;',
		'    padding: 0.6rem 1.15rem 0.7rem;',
		'    border-radius: 0.8rem;',
		'    background: rgba(15, 15, 15, 0.86);',
		'    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.4), inset 0 0 0 2px var(--cp-tip-accent), 0 0 26px var(--cp-tip-glow);',
		'    opacity: 0;',
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
		'#copilotTechToast .cp-toast.cp-toast-in { animation: copilotToastIn 420ms cubic-bezier(.2, 1.5, .4, 1) forwards; }',
		'#copilotTechToast .cp-toast.cp-toast-out { animation: copilotToastOut 440ms ease-in forwards; }',
		'@keyframes copilotToastIn {',
		'    0% { opacity: 0; transform: translateY(14px) scale(0.82); }',
		'    60% { opacity: 1; transform: translateY(-3px) scale(1.05); }',
		'    100% { opacity: 1; transform: none; }',
		'}',
		'@keyframes copilotToastOut {',
		'    0% { opacity: 1; transform: none; }',
		'    100% { opacity: 0; transform: translateY(-22px) scale(0.96); }',
		'}',
		'.cp-stage-combo {',
		'    display: none;',
		'    margin-left: 0.8em;',
		'    padding: 0.06em 0.55em;',
		'    border-radius: 999px;',
		'    background: rgba(255, 213, 74, 0.14);',
		'    color: #ffd54a;',
		'    font-size: 0.72em;',
		'    font-weight: bold;',
		'    vertical-align: middle;',
		'}',
		'.cp-stage-combo.cp-combo-on { display: inline-block; }',
		'.cp-stage-combo.cp-combo-hot {',
		'    animation: copilotComboPulse 1.3s ease-in-out infinite;',
		'}',
		'@keyframes copilotComboPulse {',
		'    0%, 100% { box-shadow: 0 0 0 rgba(255, 213, 74, 0); }',
		'    50% { box-shadow: 0 0 12px rgba(255, 213, 74, 0.7); }',
		'}',
		'#copilotAchToast {',
		'    position: fixed;',
		'    left: 50%;',
		'    top: 36%;',
		'    z-index: 46;',
		'    display: flex;',
		'    flex-direction: column;',
		'    align-items: center;',
		'    padding: 0.8rem 1.7rem 0.9rem;',
		'    border-radius: 1rem;',
		'    background: linear-gradient(160deg, rgba(36, 24, 56, 0.96), rgba(12, 12, 16, 0.94));',
		'    box-shadow: 0 0 0 2px var(--cp-ach-accent), 0 0 32px var(--cp-ach-glow), 0 18px 52px rgba(0, 0, 0, 0.55);',
		'    font-family: Arial, sans-serif;',
		'    color: #fff;',
		'    pointer-events: none;',
		'    opacity: 0;',
		'    transform: translate(-50%, -50%) scale(0.86);',
		'}',
		'.cp-ach-kicker {',
		'    font-size: 0.72rem;',
		'    letter-spacing: 0.28em;',
		'    text-transform: uppercase;',
		'    color: var(--cp-ach-accent);',
		'    margin-bottom: 0.25rem;',
		'}',
		'.cp-ach-label {',
		'    font-size: 2.05rem;',
		'    font-weight: bold;',
		'    letter-spacing: 0.03em;',
		'    color: var(--cp-ach-accent);',
		'    text-shadow: 0 0 20px var(--cp-ach-glow);',
		'    white-space: nowrap;',
		'}',
		'.cp-ach-desc {',
		'    margin-top: 0.25rem;',
		'    font-size: 0.95rem;',
		'    color: rgba(255, 255, 255, 0.86);',
		'    white-space: nowrap;',
		'}',
		'.cp-ach-more {',
		'    display: none;',
		'    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);',
		'    column-gap: 1rem;',
		'    row-gap: 0.16rem;',
		'    max-width: 34rem;',
		'    margin-top: 0.55rem;',
		'    padding-top: 0.45rem;',
		'    border-top: 1px solid rgba(255, 255, 255, 0.16);',
		'}',
		'#copilotAchToast.cp-ach-multi .cp-ach-more { display: grid; }',
		'.cp-ach-more-row {',
		'    font-size: 0.92rem;',
		'    line-height: 1.35;',
		'    color: rgba(255, 255, 255, 0.92);',
		'    white-space: nowrap;',
		'    overflow: hidden;',
		'    text-overflow: ellipsis;',
		'}',
		'.cp-ach-more-star {',
		'    margin-right: 0.4rem;',
		'    color: var(--cp-ach-accent);',
		'}',
		'.cp-ach-more-name { font-weight: bold; }',
		'.cp-ach-more-rest { opacity: 0.65; }',
		'.cp-ach-badge {',
		'    position: absolute;',
		'    top: -0.72rem;',
		'    right: -0.72rem;',
		'    display: none;',
		'    align-items: center;',
		'    justify-content: center;',
		'    min-width: 1.8rem;',
		'    height: 1.8rem;',
		'    padding: 0 0.45rem;',
		'    border-radius: 999px;',
		'    background: var(--cp-ach-accent);',
		'    color: #1a1024;',
		'    font-size: 0.95rem;',
		'    font-weight: bold;',
		'    box-shadow: 0 0 16px var(--cp-ach-glow);',
		'}',
		'#copilotAchToast.cp-ach-multi .cp-ach-badge { display: flex; }',
		'#copilotAchToast.cp-ach-multi {',
		'    box-shadow: 0 0 0 3px var(--cp-ach-accent), 0 0 56px var(--cp-ach-glow), 0 18px 52px rgba(0, 0, 0, 0.58);',
		'}',
		'@keyframes copilotAchRowIn {',
		'    0% { opacity: 0; transform: translateY(6px); }',
		'    100% { opacity: 1; transform: none; }',
		'}',
		'.cp-ach-multi .cp-ach-more-row {',
		'    animation: copilotAchRowIn 320ms ease-out both;',
		'}',
		'html.m .cp-ach-label { font-size: 1.6rem; }',
		'html.m .cp-ach-desc { font-size: 0.8rem; }',
		'html.m .cp-ach-more { grid-template-columns: minmax(0, 1fr); max-width: 20rem; }',
		'#copilotAchToast.cp-ach-in {',
		'    animation: copilotAchIn 560ms cubic-bezier(.2, 1.5, .4, 1) forwards;',
		'}',
		'#copilotAchToast.cp-ach-out {',
		'    animation: copilotAchOut 480ms ease-in forwards;',
		'}',
		'@keyframes copilotAchIn {',
		'    0% { opacity: 0; transform: translate(-50%, -50%) scale(0.72); }',
		'    55% { opacity: 1; transform: translate(-50%, -52%) scale(1.08); }',
		'    100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }',
		'}',
		'@keyframes copilotAchOut {',
		'    0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }',
		'    100% { opacity: 0; transform: translate(-50%, -62%) scale(0.94); }',
		'}',
		'#copilotAchPanel {',
		'    margin-top: 0.6rem;',
		'    padding-top: 0.5rem;',
		'    border-top: 1px solid rgba(255, 255, 255, 0.15);',
		'    font-size: 0.85em;',
		'}',
		'.cp-ach-panel-title {',
		'    flex: 1;',
		'    color: #ffd54a;',
		'    font-weight: bold;',
		'}',
		'.cp-ach-row {',
		'    display: flex;',
		'    justify-content: space-between;',
		'    gap: 0.8rem;',
		'    opacity: 0.6;',
		'    cursor: pointer;',
		'}',
		'.cp-ach-row.cp-ach-done { opacity: 1; color: #ffd54a; }',
		'.cp-ach-prog { font-family: Monospace; }',
		'.cp-ach-panel-head {',
		'    display: flex;',
		'    justify-content: space-between;',
		'    align-items: center;',
		'    gap: 0.8rem;',
		'}',
		'.cp-ach-browse {',
		'    color: #8fc8ff;',
		'    cursor: pointer;',
		'    white-space: nowrap;',
		'    font-size: 0.9em;',
		'    padding: 0.08rem 0.55rem;',
		'    border: 1px solid rgba(143, 200, 255, 0.45);',
		'    border-radius: 999px;',
		'}',
		'.cp-ach-browse:hover { text-decoration: underline; }',
		'.cp-ach-summary {',
		'    font-family: Monospace;',
		'    font-size: 0.85em;',
		'    color: rgba(255, 255, 255, 0.62);',
		'    white-space: nowrap;',
		'}',
		'#copilotAchBook {',
		'    position: fixed;',
		'    inset: 0;',
		'    z-index: 60;',
		'    display: none;',
		'    font-family: Arial, sans-serif;',
		'}',
		'#copilotAchBook.cp-book-open { display: block; }',
		'.cp-book-backdrop {',
		'    position: absolute;',
		'    inset: 0;',
		'    background: rgba(0, 0, 0, 0.62);',
		'}',
		'.cp-book-window {',
		'    position: absolute;',
		'    left: 50%;',
		'    top: 50%;',
		'    transform: translate(-50%, -50%);',
		'    width: min(92vw, 880px);',
		'    height: min(86vh, 720px);',
		'    display: flex;',
		'    flex-direction: column;',
		'    overflow: hidden;',
		'    border-radius: 14px;',
		'    background: linear-gradient(160deg, #241838, #0c0c10);',
		'    color: #fff;',
		'    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1), 0 24px 80px rgba(0, 0, 0, 0.7);',
		'}',
		'.cp-book-head {',
		'    display: flex;',
		'    align-items: center;',
		'    gap: 0.8rem;',
		'    padding: 0.75rem 1rem;',
		'    border-bottom: 1px solid rgba(255, 255, 255, 0.1);',
		'}',
		'.cp-book-title { flex: 1; font-size: 1.15rem; font-weight: bold; color: #ffd54a; }',
		'.cp-book-count { font-family: Monospace; color: rgba(255, 255, 255, 0.7); }',
		'.cp-book-reset {',
		'    cursor: pointer;',
		'    padding: 0.18rem 0.6rem;',
		'    border-radius: 999px;',
		'    background: rgba(255, 90, 90, 0.14);',
		'    color: #ff9d9d;',
		'    font-size: 0.82rem;',
		'}',
		'.cp-book-reset:hover { background: rgba(255, 90, 90, 0.26); color: #ffc4c4; }',
		'.cp-book-close {',
		'    cursor: pointer;',
		'    padding: 0 0.25rem;',
		'    font-size: 1.6rem;',
		'    line-height: 1;',
		'    opacity: 0.75;',
		'}',
		'.cp-book-close:hover { opacity: 1; }',
		'.cp-book-tabs { display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 0.6rem 1rem 0; }',
		'.cp-book-tab {',
		'    cursor: pointer;',
		'    padding: 0.2rem 0.65rem;',
		'    border-radius: 999px;',
		'    background: rgba(255, 255, 255, 0.08);',
		'    color: rgba(255, 255, 255, 0.65);',
		'    font-size: 0.9rem;',
		'}',
		'.cp-book-tab.cp-book-tab-on {',
		'    background: rgba(255, 213, 74, 0.2);',
		'    color: #ffd54a;',
		'    box-shadow: inset 0 0 0 1px rgba(255, 213, 74, 0.5);',
		'}',
		'.cp-book-grid {',
		'    flex: 1;',
		'    min-height: 0;',
		'    overflow: auto;',
		'    overscroll-behavior: contain;',
		'    padding: 0.7rem 1rem 1.2rem;',
		'    display: grid;',
		'    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));',
		'    gap: 0.65rem;',
		'    align-content: start;',
		'}',
		'.cp-book-group-title {',
		'    grid-column: 1 / -1;',
		'    margin-top: 0.35rem;',
		'    color: rgba(255, 255, 255, 0.55);',
		'    font-size: 0.78rem;',
		'    letter-spacing: 0.12em;',
		'    text-transform: uppercase;',
		'}',
		'.cp-book-card {',
		'    position: relative;',
		'    cursor: pointer;',
		'    padding: 0.65rem 0.75rem 2rem;',
		'    border-radius: 0.7rem;',
		'    background: rgba(255, 255, 255, 0.06);',
		'    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);',
		'    opacity: 0.62;',
		'    transition: opacity 0.15s, box-shadow 0.15s, transform 0.15s;',
		'}',
		'.cp-book-card::before {',
		'    content: "";',
		'    position: absolute;',
		'    left: 0;',
		'    top: 0;',
		'    bottom: 0;',
		'    width: 3px;',
		'    background: var(--cp-card-accent);',
		'    opacity: 0.75;',
		'}',
		'.cp-book-card:hover {',
		'    opacity: 1;',
		'    transform: translateY(-1px);',
		'    box-shadow: inset 0 0 0 1px var(--cp-card-accent), 0 8px 22px rgba(0, 0, 0, 0.35);',
		'}',
		'.cp-book-card.cp-book-done {',
		'    opacity: 1;',
		'    box-shadow: inset 0 0 0 1px var(--cp-card-accent), 0 0 18px rgba(255, 213, 74, 0.12);',
		'}',
		'.cp-book-card-name { font-weight: bold; color: var(--cp-card-accent); margin-bottom: 0.15rem; }',
		'.cp-book-tier {',
		'    display: inline-block;',
		'    min-width: 1.35em;',
		'    margin-right: 0.3em;',
		'    padding: 0 0.28em;',
		'    border-radius: 0.3em;',
		'    background: rgba(255, 255, 255, 0.12);',
		'    color: rgba(255, 255, 255, 0.9);',
		'    font-size: 0.72em;',
		'    text-align: center;',
		'    vertical-align: 0.08em;',
		'}',
		'.cp-book-card-desc { font-size: 0.82rem; line-height: 1.3; color: rgba(255, 255, 255, 0.75); overflow-wrap: anywhere; }',
		'.cp-book-card.cp-book-locked { opacity: 0.38; cursor: default; }',
		'.cp-book-card.cp-book-locked:hover { opacity: 0.38; transform: none; box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08); }',
		'.cp-book-card.cp-book-locked::before { opacity: 0.3; }',
		'.cp-book-card-prog {',
		'    position: absolute;',
		'    right: 0.7rem;',
		'    bottom: 0.55rem;',
		'    font-family: Monospace;',
		'    font-size: 0.8rem;',
		'    color: rgba(255, 255, 255, 0.75);',
		'}',
		'html.m .cp-book-window { width: 94vw; height: 88vh; }',
		'html.m .cp-book-grid { grid-template-columns: 1fr; }'
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

	/** the six centre stickers keep the hardware frame */
	function smartCenterKey(facelet) {
		if (!facelet || facelet.length != 54) {
			return '';
		}
		return facelet.charAt(4) + facelet.charAt(13) + facelet.charAt(22) +
			facelet.charAt(31) + facelet.charAt(40) + facelet.charAt(49);
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

	function isBlindScramble() {
		return /^(333ni|444bld|555bld|r3ni)$/.test(
			kernel.getProp('scrType', '333'));
	}

	function solveMethod(scrType) {
		if (stagePlan() == 'roux' || scrType == 'roux' ||
				hintMethod() == 'roux') {
			return 'roux';
		}
		if (/^(eoline|eocross)$/.test(scrType)) {
			return 'zz';
		}
		if (hintMethod() == 'lbl' || /corner/i.test(scrType)) {
			return 'lbl';
		}
		return 'cfop';
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
		stageBar.append(comboBadge);
		refreshComboBadge();
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

	/**
	 * Short WebAudio chime for a technique tip or an achievement.  The
	 * context is created on the first user gesture, so the browser's autoplay
	 * policy cannot block it; all failures stay silent.
	 */
	function playTipSound(cat, kind, count) {
		if (!tipAudio || !kernel.getProp('cpTipSound', true)) {
			return;
		}
		var now = Date.now();
		if (now - tipSoundAt < 220) {
			return;
		}
		tipSoundAt = now;
		try {
			if (tipAudio.state == 'suspended' && tipAudio.resume) {
				tipAudio.resume();
			}
		} catch (e) {}
		var t0 = tipAudio.currentTime + 0.01;
		function tone(freq, start, dur, peak) {
			var osc = tipAudio.createOscillator();
			var gain = tipAudio.createGain();
			osc.type = 'square';
			osc.frequency.setValueAtTime(freq, start);
			gain.gain.setValueAtTime(0.0001, start);
			gain.gain.linearRampToValueAtTime(peak, start + 0.006);
			gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
			osc.connect(gain);
			gain.connect(tipAudioGain);
			osc.start(start);
			osc.stop(start + dur + 0.01);
		}
		if (kind == 'ach') {
			// a short classic "level up" arpeggio
			var arp = [523.25, 659.25, 783.99, 1046.5];
			if (count > 1) {
				arp.push(1318.5);
			}
			if (count > 3) {
				arp.push(1567.98);
			}
			for (var i = 0; i < arp.length; i++) {
				tone(arp[i], t0 + i * 0.08,
					i == arp.length - 1 ? 0.32 : 0.085, 0.16);
			}
			return;
		}
		// coin-style two-note blip, transposed by tip rarity
		var base = cat == 'gold' ? 1046.5 : (cat == 'blue' ? 880 : 987.77);
		tone(base, t0, 0.06, 0.13);
		tone(base * 1.3348, t0 + 0.06, 0.13, 0.13);
	}

	function showTechToast(ach, silent) {
		var parts = techParts(ach);
		var el = $('<div class="cp-toast"/>')
			.attr('data-key', ach.key)
			.append(
				$('<div class="cp-toast-star">★</div>'),
				$('<div class="cp-toast-label"/>').text(parts.label),
				$('<div class="cp-toast-desc"/>').text(parts.desc)
			);
		el[0].style.setProperty('--cp-tip-accent', techAccent(ach.cat));
		el[0].style.setProperty('--cp-tip-glow', techGlow(ach.cat));
		var item = {
			key: ach.key,
			el: el,
			tid: 0,
			hold: ach.hold || (/^tps/.test(ach.key) ? 1100 :
				(ach.key.indexOf('flow') == 0 ? 750 : 1500))
		};
		techToast.append(el);
		techToastItems.push(item);
		if (!silent) {
			playTipSound(ach.cat, 'tech');
		}
		// Every tip that lands together stays visible; only drop filler cards
		// when more than four would be on screen at once.
		while (techToastItems.length > 4) {
			var drop = -1;
			for (var j = 0; j < techToastItems.length; j++) {
				if (/^tps/.test(techToastItems[j].key)) {
					drop = j;
					break;
				}
			}
			if (drop < 0) {
				for (j = 0; j < techToastItems.length; j++) {
					if (techToastItems[j].key.indexOf('flow') == 0) {
						drop = j;
						break;
					}
				}
			}
			var old = techToastItems.splice(drop >= 0 ? drop : 0, 1)[0];
			if (old.tid) {
				clearTimeout(old.tid);
			}
			old.el.remove();
		}
		positionTechToast();
		techToast.addClass('cp-toast-in');
		void el[0].offsetWidth;
		el.addClass('cp-toast-in');
		item.tid = setTimeout(function() {
			el.removeClass('cp-toast-in').addClass('cp-toast-out');
			item.tid = setTimeout(function() {
				item.tid = 0;
				for (var i = 0; i < techToastItems.length; i++) {
					if (techToastItems[i] === item) {
						techToastItems.splice(i, 1);
						break;
					}
				}
				el.remove();
				if (!techToastItems.length) {
					techToast.removeClass('cp-toast-in cp-toast-out');
				}
			}, 460);
		}, item.hold);
	}

	function hideTechToast() {
		for (var i = 0; i < techToastItems.length; i++) {
			var item = techToastItems[i];
			if (item.tid) {
				clearTimeout(item.tid);
			}
			item.el.remove();
		}
		techToastItems.length = 0;
		techToast.removeClass('cp-toast-in cp-toast-out');
	}

	// ------------------------------------------------------- achievements

	function achAccent(cat) {
		return cat == 'purple' ? '#c79bff' : (cat == 'blue' ? '#8fc8ff' : '#ffd54a');
	}

	function achGlow(cat) {
		return cat == 'purple' ? 'rgba(199, 155, 255, 0.4)' :
			(cat == 'blue' ? 'rgba(143, 200, 255, 0.35)' : 'rgba(255, 213, 74, 0.45)');
	}

	function achMeta(key) {
		return ACH_META[key] || ['', key, '', '', 'gold'];
	}

	function achDef(key) {
		var defs = achTracker.catalog();
		for (var i = 0; i < defs.length; i++) {
			if (defs[i].key == key) {
				return defs[i];
			}
		}
		return null;
	}

	function achDescText(def) {
		var desc = L(def.descKey, def.desc || '');
		if (def.param !== undefined) {
			var vals = def.param instanceof Array ? def.param : [def.param];
			var n = 0;
			desc = desc.replace(/%s/g, function() {
				return vals[Math.min(n++, vals.length - 1)];
			});
		}
		return desc;
	}

	function achHidden(key) {
		var def = achDef(key);
		if (!def) {
			return false;
		}
		if (def.secret && def.tier == 1 &&
				!achTracker.state().unlocked[key]) {
			return true;
		}
		return !!(def.prev && !achTracker.state().unlocked[def.prev]);
	}

	function achTotals() {
		var total = 0;
		var done = 0;
		achTracker.catalog().forEach(function(def) {
			var pr = achTracker.progress(def.key);
			total++;
			if (pr.cur >= pr.target) {
				done++;
			}
		});
		return {done: done, total: total};
	}

	function achSummaryText(totals) {
		return L('COPILOT_ACH_SUMMARY', 'Collected %s/%s')
			.replace('%s', totals.done)
			.replace('%s', totals.total);
	}

	/**
	 * Presentation for an achievement registered at runtime.  `meta` uses
	 * the same [nameKey, fallback, descKey, fallback, color] shape as
	 * ACH_META, so custom achievements can be added without touching the UI.
	 */
	function registerAchievementUi(key, meta) {
		if (!key || !meta) {
			return;
		}
		ACH_META[key] = meta;
		buildAchPanel();
		refreshAchPanel();
		if (achBookGrid) {
			renderAchBook();
		}
	}

	/** unified presentation info for meta achievements and technique tips */
	function achUiInfo(key) {
		var def = achDef(key);
		var meta = ACH_META[key];
		if (meta) {
			return {
				name: L(meta[0], meta[1]),
				desc: L(meta[2], meta[3]),
				color: meta[4] || 'gold',
				tip: false,
				def: def
			};
		}
		if (def && def.titleKey) {
			return {
				name: L(def.titleKey, def.title || key),
				desc: achDescText(def),
				color: def.color || 'gold',
				tip: false,
				def: def
			};
		}
		if (TECH_TEXT[key]) {
			var parts = techParts({
				key: key,
				label: TECH_LABELS[key] || key,
				name: ''
			});
			return {
				name: parts.label || TECH_LABELS[key] || key,
				desc: parts.desc,
				color: TECH_COLORS[key] || 'green',
				tip: true,
				def: def
			};
		}
		return {name: key, desc: '', color: 'green', tip: false, def: def};
	}

	var achExtraMax = 12;

	function showAchGroup(keys, preview) {
		if (!keys || !keys.length) {
			return;
		}
		var infos = [];
		for (var i = 0; i < keys.length && i <= achExtraMax; i++) {
			infos.push(achUiInfo(keys[i]));
		}
		var info = infos[0];
		if (!preview) {
			playTipSound(info.color, 'ach', keys.length);
		}
		if (preview) {
			achKicker.text(L('COPILOT_ACH_PREVIEW', 'ACHIEVEMENT PREVIEW'));
		} else if (keys.length > 1) {
			achKicker.text(L('COPILOT_ACH_UNLOCK_MULTI',
				'%s ACHIEVEMENTS UNLOCKED').replace('%s', keys.length));
		} else {
			achKicker.text(L('COPILOT_ACH_UNLOCK', 'ACHIEVEMENT UNLOCKED'));
		}
		achLabel.text(info.name);
		achDesc.text(info.desc);
		achMore.empty();
		for (i = 1; i < infos.length; i++) {
			achMore.append($('<div class="cp-ach-more-row"/>').css(
				'animation-delay', ((i - 1) * 55) + 'ms'
			).append(
				$('<span class="cp-ach-more-star"/>').text('★'),
				$('<span class="cp-ach-more-name"/>').text(infos[i].name)
			));
		}
		if (keys.length > achExtraMax + 1) {
			achMore.append($('<div class="cp-ach-more-row cp-ach-more-rest"/>')
				.text('+' + (keys.length - achExtraMax - 1)));
		}
		achBadge.text('×' + keys.length);
		achToast.toggleClass('cp-ach-multi', keys.length > 1);
		achToast[0].style.setProperty('--cp-ach-accent', achAccent(info.color));
		achToast[0].style.setProperty('--cp-ach-glow', achGlow(info.color));
		achToast.toggleClass('cp-ach-preview', !!preview);
		achToast.removeClass('cp-ach-out cp-ach-in');
		void achToast[0].offsetWidth;
		achToast.addClass('cp-ach-in');
	}

	function scheduleAchHide(hold) {
		if (achTid) {
			clearTimeout(achTid);
		}
		achTid = setTimeout(function() {
			achToast.removeClass('cp-ach-in').addClass('cp-ach-out');
			achTid = setTimeout(function() {
				achTid = 0;
				achToast.removeClass('cp-ach-out');
			}, 480);
		}, hold);
	}

	function hideAchToast() {
		if (achTid) {
			clearTimeout(achTid);
			achTid = 0;
		}
		achToast.removeClass('cp-ach-in cp-ach-out cp-ach-multi cp-ach-preview');
		achMore.empty();
	}

	function buildAchPanel() {
		var head = $('<div class="cp-ach-panel-head"/>');
		browseAchBtn.text(L('COPILOT_ACH_BROWSE', 'Browse all'));
		head.append(
			$('<div class="cp-ach-panel-title"/>').text(L('COPILOT_ACH_TITLE', 'Achievements')),
			achSummary,
			browseAchBtn
		);
		achPanel.empty().append(head);
		achRows = {};
		achTracker.list().forEach(function(key) {
			var meta = achMeta(key);
			var row = $('<div class="cp-ach-row"/>');
			var prog = $('<span class="cp-ach-prog"/>');
			row.append($('<span class="cp-ach-name"/>').text(L(meta[0], meta[1])), prog);
			achRows[key] = {row: row, prog: prog};
			achPanel.append(row);
			row.on('click', function() {
				previewAch(key);
			});
		});
		var summary = achSummaryText(achTotals());
		achSummary.text(summary);
		if (toolAchSpan) {
			toolAchSpan.text(summary);
		}
		$('.cp-ach-tool-summary').text(summary);
		browseAchBtn.off('click').on('click', openAchBook);
	}

	function refreshAchPanel() {
		var missing = false;
		achTracker.list().forEach(function(key) {
			if (!achRows[key]) {
				missing = true;
			}
		});
		if (missing) {
			// a custom achievement was registered after the panel was built
			buildAchPanel();
			return;
		}
		achTracker.list().forEach(function(key) {
			if (!achRows[key]) {
				return;
			}
			var unlocked = !!achTracker.state().unlocked[key];
			var pr = achTracker.progress(key);
			achRows[key].row.toggleClass('cp-ach-done', unlocked);
			achRows[key].prog.text(unlocked ? '✓' : pr.cur + '/' + pr.target);
		});
		var summary = achSummaryText(achTotals());
		achSummary.text(summary);
		if (toolAchSpan) {
			toolAchSpan.text(summary);
		}
		$('.cp-ach-tool-summary').text(summary);
	}

	function refreshComboBadge() {
		var hot = achTracker.state().hot;
		var on = hot >= 2;
		comboBadge.toggleClass('cp-combo-on', on)
			.toggleClass('cp-combo-hot', hot >= 3)
			.text(on ? L('COPILOT_COMBO', 'Combo ×%s').replace('%s', hot) : '');
	}

	function loadAch() {
		var raw = kernel.getProp('cpAch', '{}');
		try {
			achTracker.load(JSON.parse(raw));
		} catch (e) {
			achTracker.load(null);
		}
		refreshAchPanel();
		refreshComboBadge();
	}

	function saveAch() {
		try {
			kernel.setProp('cpAch', JSON.stringify(achTracker.serialize()));
		} catch (e) {}
	}

	function resetAchievements(force) {
		if (!force && !confirm(L('COPILOT_ACH_RESET_CONFIRM',
				'Clear all Copilot achievements and start over?'))) {
			return false;
		}
		achTracker.reset();
		saveAch();
		hideAchToast();
		refreshAchPanel();
		refreshComboBadge();
		if (achBook && achBook.hasClass('cp-book-open')) {
			renderAchBook();
		}
		return true;
	}

	// ---------------------------------------------------- achievement browser

	function achGroupLabel(group) {
		if (group == 'milestone') {
			return L('COPILOT_ACH_GROUP_MILESTONE', 'Milestones');
		}
		if (group == 'luck') {
			return L('COPILOT_ACH_GROUP_LUCK', 'Luck');
		}
		if (group == 'fun') {
			return L('COPILOT_ACH_GROUP_FUN', 'Style');
		}
		if (group == 'breadth') {
			return L('COPILOT_ACH_GROUP_BREADTH', 'Methods');
		}
		if (group == 'special') {
			return L('COPILOT_ACH_GROUP_SPECIAL', 'Special');
		}
		if (group == 'secret') {
			return L('COPILOT_ACH_GROUP_SECRET', 'Secrets');
		}
		if (group == 'technique') {
			return L('COPILOT_ACH_GROUP_TECH', 'Advanced techniques');
		}
		if (group == 'combo') {
			return L('COPILOT_ACH_GROUP_COMBO', 'Combos');
		}
		if (group == 'speed') {
			return L('COPILOT_ACH_GROUP_SPEED', 'Speed');
		}
		return L('COPILOT_ACH_GROUP_CHALLENGE', 'Challenges');
	}

	function previewAch(key) {
		if (!key || achHidden(key)) {
			return;
		}
		var info = achUiInfo(key);
		// the browser sits above the normal overlays, so lift the preview
		achToast.css('z-index', 70);
		techToast.css('z-index', 70);
		if (info.tip) {
			showTechToast({
				key: key,
				label: TECH_LABELS[key] || key,
				name: '',
				cat: TECH_COLORS[key] || 'green',
				req: false,
				hold: 1700
			}, true);
			return;
		}
		if (achTid) {
			clearTimeout(achTid);
			achTid = 0;
		}
		showAchGroup([key], true);
		scheduleAchHide(2300);
	}

	function renderAchBook() {
		if (!achBook) {
			return;
		}
		achBookTabs.find('.cp-book-tab').each(function() {
			$(this).toggleClass('cp-book-tab-on',
				$(this).attr('data-filter') == achBookFilter);
		});
		achBookGrid.empty();
		var defs = achTracker.catalog();
		var totals = achTotals();
		achBookCount.text(totals.done + '/' + totals.total);
		var groups = ['milestone', 'luck', 'fun', 'breadth', 'special',
			'secret', 'technique', 'combo', 'speed', 'challenge'];
		if (achBookFilter != 'all') {
			groups = [achBookFilter];
		}
		groups.forEach(function(group) {
			var list = defs.filter(function(def) {
				return (def.group || 'challenge') == group;
			});
			if (!list.length) {
				return;
			}
			achBookGrid.append(
				$('<div class="cp-book-group-title"/>').text(achGroupLabel(group))
			);
			list.forEach(function(def) {
				var info = achUiInfo(def.key);
				var pr = achTracker.progress(def.key);
				var ok = pr.cur >= pr.target;
				var hidden = achHidden(def.key);
				var name = $('<div class="cp-book-card-name"/>');
				if (def.tier) {
					name.append(
						$('<span class="cp-book-tier"/>').text(
							(['I', 'II', 'III'][def.tier - 1] || def.tier) + ' · '
						)
					);
				}
				name.append($('<span/>').text(hidden ? '???' : info.name));
				achBookGrid.append(
					$('<div class="cp-book-card"/>')
						.attr('data-key', def.key)
						.toggleClass('cp-book-done', ok)
						.toggleClass('cp-book-tip', !!info.tip)
						.toggleClass('cp-book-locked', hidden)
						.css('--cp-card-accent', achAccent(info.color))
						.append(
							name,
							$('<div class="cp-book-card-desc"/>').text(
								hidden ? '???' : info.desc
							),
							$('<div class="cp-book-card-prog"/>').text(
								hidden ? '' : (ok ? '✓' : pr.cur + '/' + pr.target)
							)
						)
				);
			});
		});
	}

	function ensureAchBook() {
		if (achBook) {
			return;
		}
		achBook = $('<div id="copilotAchBook"/>');
		var backdrop = $('<div class="cp-book-backdrop"/>');
		var win = $('<div class="cp-book-window"/>');
		var head = $('<div class="cp-book-head"/>');
		achBookCount = $('<span class="cp-book-count"/>');
		var reset = $('<span class="cp-book-reset click"/>')
			.attr('title', L('COPILOT_ACH_RESET', 'Reset all'))
			.text(L('COPILOT_ACH_RESET', 'Reset all'));
		var close = $('<span class="cp-book-close click"/>')
			.attr('title', L('COPILOT_ACH_CLOSE', 'Close'))
			.html('&times;');
		head.append(
			$('<div class="cp-book-title"/>').text(L('COPILOT_ACH_BROWSE', 'Achievements')),
			achBookCount,
			reset,
			close
		);
		achBookTabs = $('<div class="cp-book-tabs"/>');
		[
			['all', L('COPILOT_ACH_ALL', 'All')],
			['milestone', L('COPILOT_ACH_GROUP_MILESTONE', 'Milestones')],
			['luck', L('COPILOT_ACH_GROUP_LUCK', 'Luck')],
			['fun', L('COPILOT_ACH_GROUP_FUN', 'Style')],
			['breadth', L('COPILOT_ACH_GROUP_BREADTH', 'Methods')],
			['special', L('COPILOT_ACH_GROUP_SPECIAL', 'Special')],
			['secret', L('COPILOT_ACH_GROUP_SECRET', 'Secrets')],
			['technique', L('COPILOT_ACH_GROUP_TECH', 'Techniques')],
			['combo', L('COPILOT_ACH_GROUP_COMBO', 'Combos')],
			['speed', L('COPILOT_ACH_GROUP_SPEED', 'Speed')],
			['challenge', L('COPILOT_ACH_GROUP_CHALLENGE', 'Challenges')]
		].forEach(function(tab) {
			achBookTabs.append(
				$('<span class="cp-book-tab click"/>')
					.attr('data-filter', tab[0])
					.text(tab[1])
			);
		});
		achBookGrid = $('<div class="cp-book-grid"/>');
		win.append(head, achBookTabs, achBookGrid);
		achBook.append(backdrop, win).appendTo('body');
		backdrop.on('click', closeAchBook);
		reset.on('click', function() {
			resetAchievements();
		});
		close.on('click', closeAchBook);
		achBook.on('click', '.cp-book-tab', function() {
			achBookFilter = $(this).attr('data-filter') || 'all';
			renderAchBook();
		});
		achBook.on('click', '.cp-book-card', function() {
			previewAch($(this).attr('data-key'));
		});
		$(document).on('keydown.cpachbook', function(e) {
			if (achBook.hasClass('cp-book-open') &&
					(e.key == 'Escape' || e.keyCode == 27)) {
				closeAchBook();
				e.stopPropagation();
			}
		});
	}

	function openAchBook() {
		ensureAchBook();
		achBookFilter = 'all';
		renderAchBook();
		achBook.addClass('cp-book-open');
	}

	function closeAchBook() {
		if (!achBook) {
			return;
		}
		achBook.removeClass('cp-book-open');
		achToast.css('z-index', '');
		techToast.css('z-index', '');
	}

	function solveTotalMs(times) {
		if (!times) {
			return 0;
		}
		if (typeof times[1] == 'number' && times[1] > 0) {
			return times[1];
		}
		return (typeof times[0] == 'number' && times[0] > 0) ? times[0] : 0;
	}

	function buildSolveStats(totalMs) {
		var scrType = kernel.getProp('scrType', '333');
		var bld = isBlindScramble();
		var tst = techTracker.state();
		return {
			time: totalMs,
			flow: techFlow,
			basic: techBasic,
			tps: techTpsMax,
			norot: techNorotDone,
			openMs: techOpenMs,
			openFlow: !!(techFlowCF && techFlowFF),
			lastGap: techLastGap,
			maxGap: techMaxGap,
			choke: !!(tst && tst.choke),
			rotRun: techRotMax,
			method: bld ? '' : solveMethod(scrType),
			oh: /^333oh?$/.test(scrType),
			bld: bld,
			virtual: !isSmart && techUseMoves
		};
	}

	function finishSolve(times) {
		var stats = buildSolveStats(solveTotalMs(times));
		var keys = [];
		if (stats.method == 'cfop' && !stats.bld && techSteps) {
			for (var i = 0; i < techAch.length; i++) {
				if (techAch[i].key != 'norot') {
					keys.push(techAch[i].key);
				}
			}
			// Lower TPS tiers are collected silently, so reaching TPS 8
			// also unlocks TPS 6 instead of only the highest burst.
			for (i = 0; i < techTpsHit.length; i++) {
				if (keys.indexOf(techTpsHit[i]) < 0) {
					keys.push(techTpsHit[i]);
				}
			}
		}
		var evs = achTracker.step(keys, stats);
		saveAch();
		refreshAchPanel();
		refreshComboBadge();
		if (achBook && achBook.hasClass('cp-book-open')) {
			renderAchBook();
		}
		if (evs.length) {
			var unlocked = [];
			for (var j = 0; j < evs.length; j++) {
				unlocked.push(evs[j].key);
			}
			showAchGroup(unlocked, false);
			scheduleAchHide(Math.min(8000, 2600 + (unlocked.length - 1) * 400));
		}
	}

	function onSolveComplete(times) {
		var scrType = kernel.getProp('scrType', '333');
		// relay cubes finish on relaySolve, not on time
		if (/^r\d+$/.test(scrType)) {
			return;
		}
		if (tools.getCurPuzzle() != '333') {
			return;
		}
		if (times && typeof times[0] == 'number' && times[0] < 0) {
			return;
		}
		if (!lastSolveSolved && techLastProg != 0) {
			return;
		}
		finishSolve(times);
	}

	// ------------------------------------------------- end achievements

	function refreshTechText() {
		renderTechCombo();
		for (var i = 0; i < techToastItems.length; i++) {
			var item = techToastItems[i];
			for (var j = 0; j < techAch.length; j++) {
				if (techAch[j].key == item.key) {
					var parts = techParts(techAch[j]);
					item.el.find('.cp-toast-label').text(parts.label);
					item.el.find('.cp-toast-desc').text(parts.desc);
					break;
				}
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
				req: false,
				hold: /^tps/.test(ev.key) ? 1100 :
					(ev.key.indexOf('flow') == 0 ? 750 : 1500)
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
		var now = $.now();
		var sn = {prog: prog, t: now};
		techSteps++;
		if (!techOpenMs && prog <= 4 && techSolveStart > 0) {
			techOpenMs = Math.max(1, now - techSolveStart);
		}
		if (prog > techMaxProg) {
			techMaxProg = prog;
		}
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
			if (evs[i].key.indexOf('flow') == 0) {
				techFlow++;
			}
			if (evs[i].key == 'flowcf') {
				techFlowCF = true;
			} else if (evs[i].key == 'flowff') {
				techFlowFF = true;
			}
			onTechEvent(evs[i]);
		}
		// Require that no F2L pair was solved at the start, so a nearly
		// solved start cannot fake "all four pairs + last layer".  The first
		// observed state is usually progress 7; a cross skip reports 6.
		// A smart cube has no move stream, so fall back to its raw centre
		// frame.  Some models keep that frame fixed and cannot prove a
		// rotation either way; accept the solve instead of never unlocking.
		var canNoRot = techUseMoves || (isSmart && !!techSmartCenter);
		if (prog == 0 && canNoRot && !techRotated && !techNorotDone &&
				techMaxProg >= 6) {
			techNorotDone = true;
			onTechEvent({key: 'norot'});
		}
		checkBasicPair(facelet);
	}

	/**
	 * TPS burst: turns per second over the current run of moves.  A pause
	 * longer than 350ms starts a new burst, so a fast pair of moves after
	 * a long think does not read as a high TPS.  A tip is shown once per
	 * tier, so a fast solve does not spam the toast.
	 */
	function noteTurn(t) {
		var last = techMoveTs[techMoveTs.length - 1];
		if (last) {
			var gap = t - last;
			techLastGap = gap;
			if (gap > techMaxGap) {
				techMaxGap = gap;
			}
		}
		if (last && t - last > 350) {
			techMoveTs = [];
		}
		techMoveTs.push(t);
		if (techMoveTs.length > 10) {
			techMoveTs.shift();
		}
		if (techMoveTs.length < 5) {
			return;
		}
		var span = t - techMoveTs[0];
		if (span < 200) {
			return;
		}
		var tps = (techMoveTs.length - 1) * 1000 / span;
		if (tps > techTpsMax) {
			techTpsMax = tps;
		}
		var tiers = cubeCopilot.tpsTiers(tps);
		for (var i = 0; i < tiers.length; i++) {
			var key = 'tps' + tiers[i];
			if (techTpsHit.indexOf(key) < 0) {
				techTpsHit.push(key);
			}
		}
		var tier = tiers.length ? tiers[tiers.length - 1] : 0;
		if (tier > techTpsTier) {
			techTpsTier = tier;
			// only the highest new tier gets a toast; the lower tiers are
			// collected above and unlock at the end of the solve
			onTechEvent({key: 'tps' + tier});
		}
	}

	/** exact move info from the virtual cube: rotations vs face turns */
	function onVirtualMove(move) {
		if (!stageArmed) {
			return;
		}
		if (/^[xyz](2|')?$/.test(move)) {
			techRotated = true;
			techRotRun++;
			if (techRotRun > techRotMax) {
				techRotMax = techRotRun;
			}
			return;
		}
		techRotRun = 0;
		techUseMoves = true;
		techFaceNo++;
		noteTurn($.now());
	}

	/**
	 * A three move F2L insert: the pair was already joined in the U layer
	 * three face turns before it landed.  The history is indexed by face
	 * turns, not by state notifications, so a whole-cube rotation (which
	 * repeats the same facelet) cannot make the window shorter and turn a
	 * mid-algorithm "standard" shape into a false Basic F2L tip.
	 */
	function techFaceletAt(n) {
		for (var k = techHist.length - 1; k >= 0; k--) {
			if (techHist[k].n < n) {
				break;
			}
			if (techHist[k].n == n) {
				return techHist[k].facelet;
			}
		}
		return null;
	}

	function checkBasicPair(facelet) {
		var last = techHist[techHist.length - 1];
		var changed = !last || last.facelet != facelet;
		// Virtual cubes provide exact move events.  A bluetooth cube only sends
		// states, so count each distinct state as one face turn.
		if (isSmart && changed) {
			techFaceNo++;
		}
		var mask = 0;
		try {
			mask = cubeCopilot.pairMask(facelet);
		} catch (e) {}
		var added = mask & ~techPairMask;
		if (added) {
			var target = techFaceNo - 3;
			var before = techFaceletAt(target);
			// A longer algorithm often reaches a standard shape for its last
			// three moves.  That is a conversion, not a basic case: only
			// count it if the pair was already joined before those moves.
			var before2 = techFaceletAt(target - 1);
			for (var i = 0; i < 4; i++) {
				if ((added & (1 << i)) && before &&
						cubeCopilot.basicPair(before, i) &&
						(!before2 || cubeCopilot.basicPair(before2, i))) {
					techBasic++;
					onTechEvent({key: 'basicf2l'});
					break;
				}
			}
		}
		techPairMask = mask;
		last = techHist[techHist.length - 1];
		if (!last || last.n != techFaceNo || last.facelet != facelet) {
			techHist.push({n: techFaceNo, facelet: facelet});
			if (techHist.length > 12) {
				techHist.shift();
			}
		}
	}

	function resetTech(keepToast, seedProg) {
		if (!keepToast) {
			hideTechToast();
		}
		hideAchToast();
		techSteps = 0;
		lastSolveSolved = false;
		techAch = [];
		techF2LFacelet = null;
		techPllFacelet = null;
		techFaceNo = 0;
		techHist = [];
		techPairMask = 0;
		techFlow = 0;
		techFlowCF = false;
		techFlowFF = false;
		techBasic = 0;
		techMoveTs = [];
		techTpsMax = 0;
		techTpsTier = 0;
		techTpsHit = [];
		techSolveStart = 0;
		techOpenMs = 0;
		techLastGap = 0;
		techMaxGap = 0;
		techRotRun = 0;
		techRotMax = 0;
		techRotated = false;
		techNorotDone = false;
		techUseMoves = false;
		techSmartCenter = null;
		techMaxProg = 0;
		techLastProg = -1;
		techSeedProg = (typeof seedProg == 'number') ? seedProg : -1;
		if (techSeedProg < 0 && techSeedFacelet) {
			try {
				techSeedProg = cubeutil.getProgress(techSeedFacelet, 'cf4op');
			} catch (e) {}
		}
		if (techSeedProg > techMaxProg) {
			techMaxProg = techSeedProg;
		}
		if (techSeedFacelet) {
			techHist.push({n: 0, facelet: techSeedFacelet});
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
			lastSolveSolved = false;
			resetTech();
			techSolveStart = $.now();
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
		var solved = (stageChips.length > 0 && stageIdx >= stageChips.length) ||
			techLastProg == 0;
		lastSolveSolved = solved;
		hideStageBar(solved ? 2600 : 0);
		if (!solved) {
			hideTechToast();
			hideAchToast();
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

	function onStageFacelet(facelet, raw) {
		if (!facelet || facelet.length != 54) {
			return;
		}
		if (!stageArmed) {
			// remember the scrambled state so the first move can be judged
			techSeedFacelet = facelet;
			return;
		}
		var prog = cubeutil.getProgress(facelet, 'cf4op');
		techLastProg = prog;
		// Smart cubes only send facelets, so count each update as a turn,
		// even when a stepper plan is not configured.
		if (isSmart) {
			noteTurn($.now());
			// A bluetooth cube cannot report whole-cube rotations directly.
			// When the model includes the raw centre frame, a re-orientation
			// is the only thing that can permute it.
			var ck = smartCenterKey(raw || lastGiikerFacelet);
			if (ck) {
				if (!techSmartCenter) {
					techSmartCenter = ck;
				} else if (techSmartCenter != ck) {
					techRotated = true;
				}
			}
		}
		var plan = stagePlan();
		// Technique tips are independent of the multi-phase stepper: a
		// single-phase smart/virtual solve should still report XC/ZBLL/...
		// just like the TPS tips do.  Non-CFOP and blind solves have no
		// CFOP technique progress to read.
		if (!isBlindScramble() &&
				solveMethod(kernel.getProp('scrType', '333')) == 'cfop') {
			techStep(facelet, prog);
		}
		if (!plan) {
			return;
		}
		buildStageChips(plan);
		stageFacelet = facelet;
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
			toolAchBtn = toolAchSpan = null;
			return;
		}
		fdiv.empty();
		toolStatusSpan = $('<div/>');
		toolChkEnable = $('<input type="checkbox">');
		toolBtnHint = $('<input type="button">').val(L('COPILOT_HINT', 'Hint'));
		toolBtnClear = $('<input type="button">').val(L('COPILOT_CLEAR', 'Clear'));
		toolAchBtn = $('<input type="button">').val(L('COPILOT_ACH_BROWSE', 'Achievements'));
		toolAchSpan = $('<span/>').css({'margin-left': '0.5em', 'opacity': 0.75});
		toolHintSpan = $('<div/>').css({'font-family': 'Monospace', 'white-space': 'pre-wrap'});
		fdiv.append(
			$('<div/>').css({'font-size': '0.85em', 'margin-bottom': '0.4em'}).html(
				L('COPILOT_TOOL_HELP', 'While solving on a virtual or bluetooth cube, a hint is shown when a stage makes no progress (tune it in Options > Copilot).')),
			$('<div/>').append($('<label>').append(toolChkEnable, ' ' + L('COPILOT_ENABLED', 'Enable Copilot'))),
			toolStatusSpan,
			$('<div/>').append(toolBtnHint, ' ', toolBtnClear),
			$('<div/>').css('margin-top', '0.4em').append(toolAchBtn, toolAchSpan),
			makeCustomRow(),
			toolHintSpan
		);
		toolAchBtn.click(openAchBook);
		toolAchSpan.text(achSummaryText(achTotals()));
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

	/**
	 * Dedicated Tools entry, so the achievement browser is one click away
	 * instead of being hidden at the bottom of the Copilot window.
	 */
	function achToolExec(fdiv) {
		if (fdiv === undefined) {
			return;
		}
		var btn = $('<input type="button">').val(L('COPILOT_ACH_BROWSE', 'Browse all'));
		var btnReset = $('<input type="button">').val(L('COPILOT_ACH_RESET', 'Reset all'));
		var summary = $('<span class="cp-ach-tool-summary"/>')
			.css({'margin-left': '0.5em', 'opacity': 0.75})
			.text(achSummaryText(achTotals()));
		fdiv.empty().append(
			$('<div/>').css({'font-size': '0.85em', 'margin-bottom': '0.5em'}).text(
				L('COPILOT_ACH_TOOL_HELP',
					'Browse every Copilot achievement and technique collection. Click a card to preview it.')),
			$('<div/>').append(btn, ' ', btnReset, summary)
		);
		btn.click(openAchBook);
		btnReset.click(function() {
			resetAchievements();
		});
	}

	// ------------------------------------------------------------- signal glue

	function procSignal(signal, value) {
		if (signal == 'giikerState') {
			isSmart = true;
			lastGiikerFacelet = value[0];
			var gFacelet = smartFacelet(value[0]);
			onFacelet(gFacelet, value[0]);
			onStageFacelet(gFacelet, value[0]);
		} else if (signal == 'virtualState') {
			isSmart = false;
			smartOri = null;
			onFacelet(value[0], value[0]);
			onStageFacelet(value[0]);
		} else if (signal == 'virtualMove') {
			onVirtualMove(value[0]);
		} else if (signal == 'timerStatus') {
			onTimerStatus(value);
		} else if (signal == 'relaySolve') {
			onRelaySolve(value);
			if (stageArmed && /^r\d+$/.test(kernel.getProp('scrType'))) {
				// score the cube that just finished before its stepper resets
				if (stageChips.length && stageIdx >= stageChips.length) {
					finishSolve([0, ~~value[3]]);
				}
				if (!value[4]) {
					restartStageForNextCube();
				}
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
			if (signal == 'time') {
				onSolveComplete(value);
			}
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
		kernel.regProp('copilot', 'cpTipSound', 0, L('COPILOT_TIP_SOUND', 'Tip sound'), [true], 1);
		kernel.regProp('copilot', 'cpRelayTarget', ~2, L('COPILOT_RELAY_TARGET', '3x3 relay target per cube (s, 0 = off)'), [0, 0, 999], 1);
		kernel.regProp('copilot', 'cpLlmode', 1, L('COPILOT_LLMODE', 'Last-layer method'), ['op', ['op', 'coll', 'zbll', 'alll', 'off'], [
			L('COPILOT_LL_OP', 'OLL + PLL'),
			L('COPILOT_LL_COLL', 'COLL + EPLL'),
			L('COPILOT_LL_ZBLL', 'ZBLL'),
			L('COPILOT_LL_ALLL', '1LLL (ALLL)'),
			L('COPILOT_LL_OFF', 'Off')
		]], 1);
		kernel.regProp('copilot', 'cpCustom', ~5, 'Copilot custom algorithms', ['{}'], 1);
		kernel.regProp('copilot', 'cpAch', ~5, 'Copilot achievements', ['{}'], 1);
	}

	$(function() {
		div.append(
			$('<div class="title"/>').append(L('COPILOT_TITLE', 'Virtual/Bluetooth Cube Copilot')),
			$('<div/>').append($('<label>').append(chkEnable, ' ' + L('COPILOT_ENABLED', 'Enable Copilot'))),
			statusSpan,
			$('<div/>').append(btnHint, ' ', btnClear),
			makeCustomRow(),
			hintSpan,
			achPanel
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
		kernel.regListener('copilot', 'virtualMove', procSignal);
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
			if (techToastItems.length) {
				positionTechToast();
			}
		});

		initProps();
		buildAchPanel();
		loadAch();
		var AudioCtx = window.AudioContext || window.webkitAudioContext;
		if (AudioCtx) {
			$.waitUser.reg(function() {
				if (tipAudio) {
					return;
				}
				try {
					tipAudio = new AudioCtx();
					tipAudioGain = tipAudio.createGain();
					tipAudioGain.gain.value = 0.7;
					tipAudioGain.connect(tipAudio.destination);
				} catch (e) {
					tipAudio = null;
					tipAudioGain = null;
				}
			});
		}

		kernel.addWindow('copilot', L('COPILOT_BUTTON', 'HINT'), div, false, false, 5);
		tools.regTool('copilot', L('TOOLS_COPILOT', 'Copilot'), toolExec);
		tools.regTool('copilotach',
			L('TOOLS_COPILOT_ACH', 'Copilot achievements'), achToolExec);
		relayGlow.appendTo('body');
		relayFlash.appendTo('body');
		stageBar.appendTo('body');
		techToast.appendTo('body');
		achToast.appendTo('body');
		hintOverlay.appendTo('body');
		chkEnable.prop('checked', isEnabled());
		refreshStatus();
		refreshStage();
		startPolling();
	});

	return {
		registerAchievementUi: registerAchievementUi,
		openAchievements: openAchBook,
		resetAchievements: function() {
			return resetAchievements(true);
		},
		onFacelet: onFacelet,
		requestHint: function() {
			if (curFacelet) {
				requestHint(curFacelet, true);
			}
		},
		clear: clearAll
	};
}, [kernel, timer, tools, cubeutil, logohint]);
