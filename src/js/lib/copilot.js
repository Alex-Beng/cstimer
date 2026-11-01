"use strict";

/**
 * Cube Copilot: stage-aware hints for 3x3 CFOP solves.
 *
 * Pure logic (no DOM) shared by the main thread and the scramble worker.
 * Input is a facelet string in csTimer order ("URFDLB").  Hints:
 *   - F2L: the shortest next pair (HTM, all 18 face turns allowed)
 *   - last layer: ALLL (1LLL) recognition through a lazily built LUT
 */
function makeCubeCopilot() {
	var ALLL = null;
	var ALLL_READY = false;

	var AUF_STR = ['', 'U', 'U2', "U'"];

	var F2L_PAIRS = [
		{corner: 4, edge: 8, name: 'FR'},
		{corner: 5, edge: 9, name: 'FL'},
		{corner: 6, edge: 10, name: 'BL'},
		{corner: 7, edge: 11, name: 'BR'}
	];

	var MOVE_NAMES = (function() {
		var ret = [];
		for (var f = 0; f < 6; f++) {
			var face = 'URFDLB'.charAt(f);
			ret.push(face, face + '2', face + "'");
		}
		return ret;
	})();

	var MOVE_CUBES = null;

	function getMoveCubes() {
		if (MOVE_CUBES) {
			return MOVE_CUBES;
		}
		MOVE_CUBES = [];
		for (var i = 0; i < MOVE_NAMES.length; i++) {
			var cc = new mathlib.CubieCube();
			cc.selfMoveStr(MOVE_NAMES[i]);
			MOVE_CUBES.push(cc);
		}
		return MOVE_CUBES;
	}

	function mul(a, b) {
		var ret = new mathlib.CubieCube();
		mathlib.CubieCube.CubeMult(a, b, ret);
		return ret;
	}

	function cubieFromFacelet(facelet) {
		var cc = new mathlib.CubieCube();
		cc.fromFacelet(facelet);
		return cc;
	}

	function isPairSolved(cc, pairIdx) {
		var pair = F2L_PAIRS[pairIdx];
		return (cc.ca[pair.corner] & 7) == pair.corner && (cc.ca[pair.corner] >> 3) == 0 &&
			(cc.ea[pair.edge] >> 1) == pair.edge && (cc.ea[pair.edge] & 1) == 0;
	}

	function isCrossSolved(cc) {
		for (var e = 4; e < 8; e++) {
			if ((cc.ea[e] >> 1) != e || (cc.ea[e] & 1) != 0) {
				return false;
			}
		}
		return true;
	}

	function solvedPairMask(cc) {
		var mask = 0;
		for (var i = 0; i < 4; i++) {
			if (isPairSolved(cc, i)) {
				mask |= 1 << i;
			}
		}
		return mask;
	}

	function isF2LDone(cc) {
		return isCrossSolved(cc) && solvedPairMask(cc) == 0xf;
	}

	function isLastLayerSolved(cc) {
		return cc.isEqual(mathlib.CubieCube.SOLVED);
	}

	/** 'cross' | 'f2l' | 'll' | 'solved' */
	function getStage(cc) {
		if (isLastLayerSolved(cc)) {
			return 'solved';
		}
		if (isF2LDone(cc)) {
			return 'll';
		}
		if (isCrossSolved(cc)) {
			return 'f2l';
		}
		return 'cross';
	}

	/**
	 * Pruning coordinate: where the pair's corner and edge pieces currently are,
	 * plus their orientation.  This is invariant under every face turn (a move
	 * maps position/orientation deterministically), which is what makes a BFS
	 * pruning table valid.  Using "which piece sits in the slot" instead would
	 * not be closed under moves and would prune away real solutions.
	 *
	 * Encoded as a small integer (576 values) so the hot search loop never
	 * builds strings.
	 */
	function pairKey(cc, pairIdx) {
		var pair = F2L_PAIRS[pairIdx];
		var cp = 0;
		var co = 0;
		for (var p = 0; p < 8; p++) {
			if ((cc.ca[p] & 7) == pair.corner) {
				cp = p;
				co = cc.ca[p] >> 3;
				break;
			}
		}
		var ep = 0;
		var eo = 0;
		for (var q = 0; q < 12; q++) {
			if ((cc.ea[q] >> 1) == pair.edge) {
				ep = q;
				eo = cc.ea[q] & 1;
				break;
			}
		}
		return (cp * 3 + co) * 24 + ep * 2 + eo;
	}

	/** facelet indices of a slot: 3 corner stickers + 2 edge stickers */
	function slotFacelets(pairIdx) {
		var pair = F2L_PAIRS[pairIdx];
		var ret = mathlib.CubieCube.cFacelet[pair.corner].slice();
		return ret.concat(mathlib.CubieCube.eFacelet[pair.edge]);
	}

	// ------------------------------------------------------------- F2L search

	// pair coordinate: (cp*3+co)*24 + ep*2 + eo  -> 576 values
	var PAIR_STATES = 576;
	var F2L_PRUN = [null, null, null, null];

	// The pair coordinate has only ~600 reachable values, so the pruning table
	// is built to full closure.  A depth-limited table would report "unknown"
	// (99) for far states and wrongly prune valid solutions.
	function buildPairPrun(pairIdx) {
		var table = new Int8Array(PAIR_STATES);
		table.fill(-1);
		var solved = new mathlib.CubieCube();
		table[pairKey(solved, pairIdx)] = 0;
		var frontier = [solved];
		var cubes = getMoveCubes();
		var depth = 0;
		while (frontier.length && depth < 32) {
			depth++;
			var next = [];
			for (var f = 0; f < frontier.length; f++) {
				for (var m = 0; m < cubes.length; m++) {
					var ns = mul(frontier[f], cubes[m]);
					var key = pairKey(ns, pairIdx);
					if (table[key] >= 0) {
						continue;
					}
					table[key] = depth;
					next.push(ns);
				}
			}
			frontier = next;
		}
		F2L_PRUN[pairIdx] = table;
		return table;
	}

	function pruneTable(pairIdx) {
		return F2L_PRUN[pairIdx] || buildPairPrun(pairIdx);
	}

	function pairDistance(pairIdx, cc) {
		var v = pruneTable(pairIdx)[pairKey(cc, pairIdx)];
		return v < 0 ? 99 : v;
	}

	/**
	 * Cross pruning coordinate: position + orientation of the four D edges.
	 * Also closed under moves, so a full-closure BFS is valid (~190k states).
	 * Encoded as a base-24 integer: 24^4 = 331776 entries.
	 */
	var CROSS_STATES = 331776;
	var CROSS_PRUN = null;
	var CROSS_EDGES = [4, 5, 6, 7];

	function crossKey(cc) {
		var ret = 0;
		for (var i = 0; i < CROSS_EDGES.length; i++) {
			var piece = CROSS_EDGES[i];
			for (var p = 0; p < 12; p++) {
				if ((cc.ea[p] >> 1) == piece) {
					ret = ret * 24 + p * 2 + (cc.ea[p] & 1);
					break;
				}
			}
		}
		return ret;
	}

	function crossPrunTable() {
		if (CROSS_PRUN) {
			return CROSS_PRUN;
		}
		var table = new Int8Array(CROSS_STATES);
		table.fill(-1);
		var solved = new mathlib.CubieCube();
		table[crossKey(solved)] = 0;
		var frontier = [solved];
		var cubes = getMoveCubes();
		var depth = 0;
		while (frontier.length && depth < 16) {
			depth++;
			var next = [];
			for (var f = 0; f < frontier.length; f++) {
				for (var m = 0; m < cubes.length; m++) {
					var ns = mul(frontier[f], cubes[m]);
					var key = crossKey(ns);
					if (table[key] >= 0) {
						continue;
					}
					table[key] = depth;
					next.push(ns);
				}
			}
			frontier = next;
		}
		CROSS_PRUN = table;
		return table;
	}

	function crossDistance(cc) {
		var v = crossPrunTable()[crossKey(cc)];
		return v < 0 ? 99 : v;
	}

	/**
	 * Lower bound for a position: the cross must end up solved and every
	 * protected pair must end up solved, so the max of their distances is
	 * admissible.
	 */
	function lowerBound(cc, pairIdx, protectMask) {
		var h = pairDistance(pairIdx, cc);
		var c = crossDistance(cc);
		if (c > h) {
			h = c;
		}
		for (var p = 0; p < 4; p++) {
			if (!(protectMask & (1 << p)) || p == pairIdx) {
				continue;
			}
			var d = pairDistance(p, cc);
			if (d > h) {
				h = d;
			}
		}
		return h;
	}

	function searchPair(cc, pairIdx, maxDepth, protectMask, deadline) {
		var moves = getMoveCubes();
		var path = [];

		function dfs(state, remaining, lastFace) {
			if (deadline && $.now() > deadline) {
				return false;
			}
			if (lowerBound(state, pairIdx, protectMask) > remaining) {
				return false;
			}
			if (remaining == 0) {
				if (!isPairSolved(state, pairIdx) || !isCrossSolved(state)) {
					return false;
				}
				for (var p = 0; p < 4; p++) {
					if ((protectMask & (1 << p)) && !isPairSolved(state, p)) {
						return false;
					}
				}
				return true;
			}
			for (var m = 0; m < moves.length; m++) {
				var face = ~~(m / 3);
				if (face == lastFace) {
					continue;
				}
				var ns = mul(state, moves[m]);
				path.push(MOVE_NAMES[m]);
				if (dfs(ns, remaining - 1, face)) {
					return true;
				}
				path.pop();
			}
			return false;
		}

		for (var depth = 1; depth <= maxDepth; depth++) {
			path = [];
			if (dfs(cc, depth, -1)) {
				return path.slice();
			}
		}
		return null;
	}

	/**
	 * Shortest next pair: complete any unsolved pair while keeping the cross and
	 * every already solved pair intact.  R2 counts as one move (HTM).
	 *
	 * options.timeLimit (ms) bounds the work.  When the budget is exhausted the
	 * best solution found so far is returned with optimal=false; otherwise the
	 * result is proven minimal (every shallower depth was exhausted).
	 */
	/**
	 * Every unsolved pair that fits in maxDepth, shortest first.  `count`
	 * limits the list (the UI can show several F2L hints at once).
	 */
	function solvePairs(facelet, options, count) {
		options = options || {};
		var maxDepth = options.maxDepth || 8;
		var deadline = options.timeLimit ? $.now() + options.timeLimit : 0;
		var cc = cubieFromFacelet(facelet);
		var solved = solvedPairMask(cc);
		if (!isCrossSolved(cc) || solved == 0xf) {
			return [];
		}
		var found = [];
		var timedOut = false;
		for (var i = 0; i < 4; i++) {
			if (solved & (1 << i)) {
				continue;
			}
			// cheap lower bound: skip pairs that provably need more than maxDepth
			if (lowerBound(cc, i, solved) > maxDepth) {
				continue;
			}
			var path = searchPair(cc, i, maxDepth, solved, deadline);
			if (!path && deadline && $.now() > deadline) {
				timedOut = true;
			}
			if (path) {
				found.push({
					pair: i,
					name: F2L_PAIRS[i].name,
					key: 'F2L/' + F2L_PAIRS[i].name + '/' + pairKey(cc, i),
					base: path.slice(),
					pre: '',
					post: '',
					moves: path,
					alg: path.join(' '),
					slot: slotFacelets(i),
					corner: F2L_PAIRS[i].corner,
					edge: F2L_PAIRS[i].edge,
					optimal: !timedOut
				});
			}
		}
		found.sort(function(a, b) {
			return a.moves.length - b.moves.length || a.pair - b.pair;
		});
		for (var f = 0; f < found.length; f++) {
			found[f].optimal = !timedOut;
		}
		return found.slice(0, Math.max(1, count || 1));
	}

	function solveNextPair(facelet, options) {
		var pairs = solvePairs(facelet, options, 1);
		return pairs.length ? pairs[0] : null;
	}

	/** shortest sequence that completes the D cross (used to guide back) */
	function solveCross(facelet, options) {
		options = options || {};
		var maxDepth = options.maxDepth || 8;
		var deadline = options.timeLimit ? $.now() + options.timeLimit : 0;
		var cc = cubieFromFacelet(facelet);
		if (isCrossSolved(cc)) {
			return [];
		}
		var moves = getMoveCubes();
		var path = [];

		function dfs(state, remaining, lastFace) {
			if (deadline && $.now() > deadline) {
				return false;
			}
			if (crossDistance(state) > remaining) {
				return false;
			}
			if (remaining == 0) {
				return isCrossSolved(state);
			}
			for (var m = 0; m < moves.length; m++) {
				var face = ~~(m / 3);
				if (face == lastFace) {
					continue;
				}
				var ns = mul(state, moves[m]);
				path.push(MOVE_NAMES[m]);
				if (dfs(ns, remaining - 1, face)) {
					return true;
				}
				path.pop();
			}
			return false;
		}

		for (var depth = 1; depth <= maxDepth; depth++) {
			path = [];
			if (dfs(cc, depth, -1)) {
				return path.slice();
			}
		}
		return null;
	}

	/**
	 * Build every pruning table up front so the first hint is not slow.
	 * Returns the number of milliseconds spent.
	 */
	function warmUp() {
		var t = $.now();
		for (var i = 0; i < 4; i++) {
			pruneTable(i);
		}
		crossPrunTable();
		buildOllLut();
		buildPllLut();
		return $.now() - t;
	}

	// ------------------------------------------------------------ ALLL / 1LLL

	/** permutation/orientation of the U layer only (ignores whole-cube rotation) */
	function llKey(cc) {
		var ret = [];
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ca[i] & 7, cc.ca[i] >> 3);
		}
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ea[i] >> 1, cc.ea[i] & 1);
		}
		return ret.join(',');
	}

	/** face-move indices actually applied by one token, in the given frame */
	function movesAppliedBy(token, ori) {
		var m = /^([URFDLB])(w?)(2|'|2')?$/.exec(token);
		if (m) {
			var axis = 'URFDLB'.indexOf(m[1]);
			var pow = powOf(m[3] || '');
			var faceIdx = m[2] ? (axis + 3) % 6 : axis;
			return [mathlib.CubieCube.rotMulM[ori][faceIdx * 3 + pow % 4 - 1]];
		}
		var s = /^([EMS])(2|'|2')?$/.exec(token);
		if (s) {
			var saxis = {S: 2, E: 3, M: 4}[s[1]];
			var spow = powOf(s[2] || '');
			return [
				mathlib.CubieCube.rotMulM[ori][saxis * 3 + (4 - spow) % 4 - 1],
				mathlib.CubieCube.rotMulM[ori][(saxis + 3) % 6 * 3 + spow % 4 - 1]
			];
		}
		return [];
	}

	function powOf(mod) {
		if (mod == "'") {
			return 3;
		}
		if (mod == '2' || mod == "2'") {
			return 2;
		}
		return 1;
	}

	/**
	 * Inverse of a token sequence as a *function*.
	 *
	 * `ori` is part of the state: a wide/slice token changes the frame used by
	 * the moves that follow, so reversing the tokens with primed moves is not
	 * the inverse.  The frame depends only on the token sequence, so record it
	 * before every token and undo the applied face moves in reverse order.
	 */
	function inverseState(tokens) {
		var probe = new mathlib.CubieCube();
		var oriBefore = new Array(tokens.length);
		for (var i = 0; i < tokens.length; i++) {
			oriBefore[i] = probe.ori;
			probe.selfMoveStr(tokens[i]);
		}
		var cc = new mathlib.CubieCube();
		var tmp = new mathlib.CubieCube();
		for (var i = tokens.length - 1; i >= 0; i--) {
			var applied = movesAppliedBy(tokens[i], oriBefore[i]);
			for (var k = applied.length - 1; k >= 0; k--) {
				var inv = ~~(applied[k] / 3) * 3 + (2 - applied[k] % 3);
				mathlib.CubieCube.CubeMult(cc, mathlib.CubieCube.moveCube[inv], tmp);
				cc.init(tmp.ca, tmp.ea);
			}
		}
		cc.ori = 0;
		return cc;
	}

	/**
	 * A last-layer key is legal only when the twist/flip sums and the two
	 * permutation parities agree.  Algs whose net effect rotates the centres
	 * (a leftover slice) would otherwise inject unreachable keys.
	 */
	function isLegalKey(key) {
		var p = key.split(',').map(Number);
		var coSum = p[1] + p[3] + p[5] + p[7];
		var eoSum = p[9] + p[11] + p[13] + p[15];
		var cp = [p[0], p[2], p[4], p[6]];
		var ep = [p[8], p[10], p[12], p[14]];
		function inv(a) {
			var c = 0;
			for (var i = 0; i < a.length; i++) {
				for (var j = i + 1; j < a.length; j++) {
					if (a[i] > a[j]) {
						c++;
					}
				}
			}
			return c % 2;
		}
		return coSum % 3 == 0 && eoSum % 2 == 0 && inv(cp) == inv(ep);
	}

	/**
	 * Build the 1LLL recognition LUT: for every base alg and pre/post AUF, the
	 * position it solves becomes the key.
	 */
	function buildAlllLut() {
		ALLL = {};
		var list = (typeof ALLL_DATA == 'undefined') ? [] : ALLL_DATA;
		var stats = {entries: list.length, slots: 0, illegal: 0};
		for (var i = 0; i < list.length; i++) {
			var entry = list[i];
			var sep = entry.indexOf('|');
			var name = entry.slice(0, sep);
			var tokens = entry.slice(sep + 1).split(/\s+/).filter(function(t) {
				return !!t;
			});
			for (var r = 0; r < 4; r++) {
				for (var p = 0; p < 4; p++) {
					var seq = [];
					if (AUF_STR[r]) {
						seq.push(AUF_STR[r]);
					}
					seq = seq.concat(tokens);
					if (AUF_STR[p]) {
						seq.push(AUF_STR[p]);
					}
					seq = simplifySeq(seq);
					var key = llKey(inverseState(seq));
					if (!isLegalKey(key)) {
						stats.illegal++;
						continue;
					}
					var display = seq.join(' ');
					if (!(key in ALLL) || display.length < ALLL[key].alg.length) {
						ALLL[key] = {name: name, alg: display, len: display.split(/\s+/).length};
					}
					stats.slots++;
				}
			}
		}
		stats.unique = Object.keys(ALLL).length;
		ALLL_READY = true;
		return stats;
	}

	function ensureAlllLut() {
		if (!ALLL_READY) {
			buildAlllLut();
		}
		return ALLL;
	}

	function alllReady() {
		return ALLL_READY;
	}

	function alllLutSize() {
		return ALLL ? Object.keys(ALLL).length : 0;
	}

	// ------------------------------------------------------------ OLL / PLL

	var OLL_LUT = null;
	var PLL_LUT = null;
	var U_CUBE = null;

	function tokensOf(str) {
		return str ? str.split(/\s+/).filter(function(t) {
			return !!t;
		}) : [];
	}

	function parseMoveToken(token) {
		var m = /^([URFDLB]w?|2-2[URFDLB]w|[EMSxyz])(2|'|2')?$/.exec(token);
		if (!m) {
			return null;
		}
		var mod = m[2] || '';
		return {base: m[1], pow: mod == '' ? 1 : (mod == "'" ? 3 : 2)};
	}

	/**
	 * Cancel adjacent turns of the same face, which is what creates the
	 * "U U'" noise when an algorithm already starts/ends with the AUF that was
	 * prepended/appended for recognition.
	 */
	function simplifySeq(tokens) {
		var out = [];
		for (var i = 0; i < tokens.length; i++) {
			var t = parseMoveToken(tokens[i]);
			if (!t) {
				out.push(tokens[i]);
				continue;
			}
			var merged = false;
			if (out.length) {
				var last = parseMoveToken(out[out.length - 1]);
				if (last && last.base == t.base) {
					var pow = (last.pow + t.pow) % 4;
					out.pop();
					if (pow) {
						out.push(t.base + (pow == 1 ? '' : (pow == 2 ? '2' : "'")));
					}
					merged = true;
				}
			}
			if (!merged) {
				out.push(t.base + (t.pow == 1 ? '' : (t.pow == 2 ? '2' : "'")));
			}
		}
		return out;
	}

	var MOVE_CANDIDATES = null;

	function moveCandidates() {
		if (MOVE_CANDIDATES) {
			return MOVE_CANDIDATES;
		}
		var ret = [];
		// plain face turns first: a wide turn has the same piece layout as a
		// face turn plus a whole-cube rotation, and deviations are normally
		// plain face turns
		for (var f = 0; f < 6; f++) {
			var face = 'URFDLB'.charAt(f);
			ret.push(face, face + '2', face + "'");
		}
		for (var g = 0; g < 6; g++) {
			var wide = 'URFDLB'.charAt(g) + 'w';
			ret.push(wide, wide + '2', wide + "'");
		}
		var slices = ['M', 'E', 'S'];
		for (var s = 0; s < slices.length; s++) {
			ret.push(slices[s], slices[s] + '2', slices[s] + "'");
		}
		MOVE_CANDIDATES = ret;
		return ret;
	}

	function cubieSig(cc) {
		return cc.ca.join(',') + '|' + cc.ea.join(',');
	}

	/**
	 * Which single turn turns `from` into `to`?  Used to build the inverse
	 * sequence when the user leaves the hinted algorithm (like the smart-cube
	 * scramble tracking re-syncing after a wrong move).
	 */
	function inferMove(fromFacelet, toFacelet) {
		if (!fromFacelet || !toFacelet) {
			return null;
		}
		var from = cubieFromFacelet(fromFacelet);
		var to = cubieFromFacelet(toFacelet);
		var target = cubieSig(to);
		var cands = moveCandidates();
		for (var i = 0; i < cands.length; i++) {
			var c = cloneCube(from);
			c.selfMoveStr(cands[i]);
			if (cubieSig(c) == target) {
				return cands[i];
			}
		}
		return null;
	}

	/** inverse of one move token (2 is self-inverse) */
	function invertToken(token) {
		var m = /^([URFDLB]w?|2-2[URFDLB]w|[EMSxyz])(2|'|2')?$/.exec(token);
		if (!m) {
			return token;
		}
		var mod = m[2] || '';
		if (mod == '2' || mod == "2'") {
			return m[1] + '2';
		}
		if (mod == "'") {
			return m[1];
		}
		return m[1] + "'";
	}

	function parseEntry(entry) {
		var sep = entry.indexOf('|');
		if (sep < 0) {
			return null;
		}
		var tokens = tokensOf(entry.slice(sep + 1));
		if (!tokens.length) {
			return null;
		}
		return {name: entry.slice(0, sep), tokens: tokens};
	}

	/** which way each of the four U corners and four U edges is twisted */
	function ollPattern(cc) {
		var ret = [];
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ca[i] >> 3);
		}
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ea[i] & 1);
		}
		return ret.join(',');
	}

	function isOriented(cc) {
		for (var i = 0; i < 4; i++) {
			if ((cc.ca[i] >> 3) != 0 || (cc.ea[i] & 1) != 0) {
				return false;
			}
		}
		return true;
	}

	function buildOllLut() {
		OLL_LUT = {};
		var list = (typeof OLL_DATA == 'undefined') ? [] : OLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			var key = ollPattern(inverseState(e.tokens));
			if (!(key in OLL_LUT)) {
				OLL_LUT[key] = {name: e.name, tokens: e.tokens};
			}
		}
		return OLL_LUT;
	}

	function buildPllLut() {
		PLL_LUT = {};
		var list = (typeof PLL_DATA == 'undefined') ? [] : PLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var key = llKey(inverseState(seq));
					if (!isLegalKey(key)) {
						continue;
					}
					var disp = seq.join(' ');
					if (!(key in PLL_LUT) || disp.length < PLL_LUT[key].alg.length) {
						PLL_LUT[key] = {
							name: e.name,
							tokens: e.tokens,
							pre: AUF_STR[pre],
							post: AUF_STR[post],
							alg: disp
						};
					}
				}
			}
		}
		return PLL_LUT;
	}

	function ensureOllLut() {
		return OLL_LUT || buildOllLut();
	}

	function ensurePllLut() {
		return PLL_LUT || buildPllLut();
	}

	function uCube() {
		if (!U_CUBE) {
			U_CUBE = new mathlib.CubieCube();
			U_CUBE.selfMoveStr('U');
		}
		return U_CUBE;
	}

	/** OLL: orientation-only LUT, so the four AUFs must be tried at lookup */
	function ollHint(cc) {
		var lut = ensureOllLut();
		var test = new mathlib.CubieCube();
		test.init(cc.ca.slice(), cc.ea.slice());
		for (var auf = 0; auf < 4; auf++) {
			if (auf) {
				test = mul(test, uCube());
			}
			var hit = lut[ollPattern(test)];
			if (hit) {
				var moves = simplifySeq(tokensOf(AUF_STR[auf]).concat(hit.tokens));
				return {
					kind: 'oll',
					name: hit.name,
					key: 'OLL/' + hit.name,
					base: hit.tokens.slice(),
					pre: AUF_STR[auf],
					post: '',
					moves: moves,
					alg: moves.join(' ')
				};
			}
		}
		return null;
	}

	function pllHint(cc) {
		var lut = ensurePllLut();
		var hit = lut[llKey(cc)];
		if (hit) {
			var moves = simplifySeq(tokensOf(hit.pre).concat(hit.tokens, tokensOf(hit.post)));
			return {
				kind: 'pll',
				name: hit.name,
				key: 'PLL/' + hit.name,
				base: hit.tokens.slice(),
				pre: hit.pre,
				post: hit.post,
				moves: moves,
				alg: moves.join(' ')
			};
		}
		// the table stores every pre/post AUF, but a rotated lookup is still
		// used as a safety net for states that only the rotation reaches
		var test = new mathlib.CubieCube();
		test.init(cc.ca.slice(), cc.ea.slice());
		for (var auf = 1; auf < 4; auf++) {
			test = mul(test, uCube());
			hit = lut[llKey(test)];
			if (hit) {
				var pre = tokensOf(AUF_STR[auf]).concat(tokensOf(hit.pre));
				var moves2 = simplifySeq(pre.concat(hit.tokens, tokensOf(hit.post)));
				return {
					kind: 'pll',
					name: hit.name,
					key: 'PLL/' + hit.name,
					base: hit.tokens.slice(),
					pre: pre.join(' '),
					post: hit.post,
					moves: moves2,
					alg: moves2.join(' ')
				};
			}
		}
		return null;
	}

	function aufHint(cc) {
		var auf = solveByAuf(cc);
		if (!auf) {
			return null;
		}
		return {
			kind: 'auf',
			name: 'AUF',
			key: '',
			base: [auf],
			pre: '',
			post: '',
			moves: [auf],
			alg: auf
		};
	}

	/** If a single U turn solves the layer, return that turn. */
	function solveByAuf(cc) {
		var AUF = ['', 'U', 'U2', "U'"];
		for (var k = 1; k < 4; k++) {
			var test = new mathlib.CubieCube();
			test.init(cc.ca.slice(), cc.ea.slice());
			for (var i = 0; i < k; i++) {
				test.selfMoveStr('U');
			}
			if (test.isEqual(mathlib.CubieCube.SOLVED)) {
				return AUF[k];
			}
		}
		return null;
	}

	/**
	 * Solve the last layer with the requested method:
	 *   'alll' (default): one algorithm from the 1LLL table
	 *   'op': two-look OLL then PLL, falling back to the 1LLL table
	 *   'off': no last-layer hint
	 */
	function solveLastLayer(facelet, mode) {
		var cc = cubieFromFacelet(facelet);
		if (!isF2LDone(cc)) {
			return null;
		}
		if (isLastLayerSolved(cc)) {
			return {kind: 'skip', name: 'skip', alg: '', moves: []};
		}
		mode = mode || 'alll';
		if (mode == 'off') {
			return {kind: 'off', name: 'off', alg: '', moves: []};
		}
		if (mode == 'op') {
			var ll;
			if (!isOriented(cc)) {
				ll = ollHint(cc);
			} else {
				ll = pllHint(cc) || aufHint(cc);
			}
			if (ll) {
				return ll;
			}
			// positions missing from the two-look sets still get a 1LLL alg
		}
		var hit = ensureAlllLut()[llKey(cc)];
		if (!hit) {
			var auf = aufHint(cc);
			if (auf) {
				return auf;
			}
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'alll',
			name: hit.name,
			key: 'ALLL/' + llKey(cc),
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	function getHint(facelet, options) {
		options = options || {};
		var cc = cubieFromFacelet(facelet);
		var stage = getStage(cc);
		if (stage == 'f2l') {
			var pairs = solvePairs(facelet, options, options.f2lCount);
			if (!pairs.length) {
				return {stage: 'f2l', name: 'no-solution'};
			}
			var pair = pairs[0];
			return {
				stage: 'f2l',
				kind: 'f2l',
				pair: pair.pair,
				name: pair.name,
				key: pair.key,
				base: pair.moves.slice(),
				pre: '',
				post: '',
				moves: pair.moves,
				alg: pair.moves.join(' '),
				slot: pair.slot,
				corner: pair.corner,
				edge: pair.edge,
				pairs: pairs
			};
		}
		if (stage == 'll') {
			var ll = solveLastLayer(facelet, options.llMode);
			if (!ll) {
				return {stage: 'll', name: 'no-alg'};
			}
			return {
				stage: 'll',
				kind: ll.kind,
				name: ll.name,
				key: ll.key,
				base: ll.base,
				pre: ll.pre,
				post: ll.post,
				moves: ll.moves,
				alg: ll.alg
			};
		}
		return {stage: stage};
	}

	// -------------------------------------------------------------- utilities

	/** facelet string normalised against whole-cube orientation */
	function normalizeFacelet(facelet) {
		return cubieFromFacelet(facelet).toFaceCube();
	}

	/** normalised facelet after each token of the algorithm */
	function algStates(facelet, tokens) {
		var cc = cubieFromFacelet(facelet);
		var ret = [];
		for (var i = 0; i < tokens.length; i++) {
			cc.selfMoveStr(tokens[i]);
			ret.push(cc.toFaceCube());
		}
		return ret;
	}

	function cloneCube(cc) {
		var ret = new mathlib.CubieCube();
		ret.init(cc.ca.slice(), cc.ea.slice());
		ret.ori = cc.ori;
		return ret;
	}

	/**
	 * Orientation-independent signature of a position.
	 *
	 * Turning the whole cube only re-expresses the same position in another
	 * frame, which is a conjugation by the rotation, so the minimum over the 24
	 * rotations is invariant under a regrip (x/y/z).
	 */
	function canonicalSigOf(cc) {
		var rots = mathlib.CubieCube.rotCube;
		var inv = mathlib.CubieCube.rotMulI[0];
		var tmp = new mathlib.CubieCube();
		var tmp2 = new mathlib.CubieCube();
		var best = null;
		for (var i = 0; i < rots.length; i++) {
			mathlib.CubieCube.CubeMult(rots[inv[i]], cc, tmp);
			mathlib.CubieCube.CubeMult(tmp, rots[i], tmp2);
			var sig = tmp2.ca.join(',') + '|' + tmp2.ea.join(',');
			if (best === null || sig < best) {
				best = sig;
			}
		}
		return best;
	}

	function canonicalSig(facelet) {
		return canonicalSigOf(cubieFromFacelet(facelet));
	}

	/**
	 * Re-express a position as seen from a rotated view (a whole-cube
	 * orientation index from mathlib's rot2str), so hints use the face names
	 * the user actually sees.  Smart cubes report a fixed frame; csTimer's
	 * "Cube orientation" setting lets the user align that frame with the grip.
	 */
	function orientFacelet(facelet, ori) {
		if (ori == null || ori < 0 || ori >= 24) {
			return facelet;
		}
		var str = mathlib.CubieCube.rot2str[ori];
		if (!str) {
			return facelet;
		}
		var cc = cubieFromFacelet(facelet);
		str.split(/\s+/).filter(function(t) {
			return !!t;
		}).forEach(function(t) {
			cc.selfMoveStr(t);
		});
		return cc.toFaceCube(null, null, null, true);
	}

	/** preferred front colour (raw face letters) after the cross is at D */
	var FRONT_PRIORITY = 'FRBLUD';

	/**
	 * Smart-cube frame for hints: once the cross is solved, put it at the
	 * bottom (this is what the CFOP stage timing is counting), then pick the
	 * front from a fixed colour order.  Before that, use the fixed order
	 * (raw U at D, raw F at F).
	 */
	function pickSmartOrientation(facelet) {
		var best = null;
		for (var r = 0; r < 24; r++) {
			var f = orientFacelet(facelet, r);
			var cc = cubieFromFacelet(f);
			if (!isCrossSolved(cc)) {
				continue;
			}
			// mirror the CFOP stage result: prefer the axis with more F2L done
			var mask = solvedPairMask(cc);
			var pairCount = 0;
			for (var p = 0; p < 4; p++) {
				if (mask & (1 << p)) {
					pairCount++;
				}
			}
			var frontScore = FRONT_PRIORITY.indexOf(f.charAt(22));
			if (frontScore < 0) {
				frontScore = 99;
			}
			var score = (4 - pairCount) * 10000 + frontScore * 100 + r;
			if (!best || score < best.score) {
				best = {
					facelet: f,
					ori: r,
					d: f.charAt(31),
					f: f.charAt(22),
					score: score,
					cross: true,
					pairCount: pairCount
				};
			}
		}
		if (best) {
			return best;
		}
		for (var r2 = 0; r2 < 24; r2++) {
			var f2 = orientFacelet(facelet, r2);
			if (f2.charAt(31) == 'U' && f2.charAt(22) == 'F') {
				return {facelet: f2, ori: r2, d: 'U', f: 'F', cross: false, pairCount: 0};
			}
		}
		return {facelet: facelet, ori: 0, d: 'U', f: 'F', cross: false, pairCount: 0};
	}

	/**
	 * Steps of an algorithm, one entry per intermediate position.  A half turn
	 * (`R2`) gets an extra entry after each quarter turn (`R` and `R'`), so a
	 * user who turns it half way is seen as following the algorithm, not as
	 * making a mistake.  `remain` is the algorithm still to run from that
	 * position.
	 */
	function algSteps(facelet, tokens) {
		var cc = cubieFromFacelet(facelet);
		var steps = [];
		for (var i = 0; i < tokens.length; i++) {
			var token = tokens[i];
			var rest = tokens.slice(i + 1);
			var half = /^([URFDLB]w?|2-2[URFDLB]w|[EMSxyz])(2|2')$/.exec(token);
			if (half) {
				var base = half[1];
				if (base.indexOf('2-2') == 0) {
					base = base.slice(3);
				}
				var cw = cloneCube(cc);
				cw.selfMoveStr(base);
				steps.push({sig: canonicalSigOf(cw), remain: [base].concat(rest)});
				var ccw = cloneCube(cc);
				ccw.selfMoveStr(base + "'");
				steps.push({sig: canonicalSigOf(ccw), remain: [base + "'"].concat(rest)});
			}
			cc.selfMoveStr(token);
			steps.push({sig: canonicalSigOf(cc), remain: rest});
		}
		return steps;
	}

	/** does `tokens` really solve the case described by `hint`? */
	function validateAlg(facelet, tokens, hint) {
		if (!tokens || !tokens.length || !hint) {
			return false;
		}
		var before = cubieFromFacelet(facelet);
		var cc = cubieFromFacelet(facelet);
		for (var i = 0; i < tokens.length; i++) {
			cc.selfMoveStr(tokens[i]);
		}
		if (hint.stage == 'f2l') {
			if (!isCrossSolved(cc) || typeof hint.pair != 'number' || !isPairSolved(cc, hint.pair)) {
				return false;
			}
			var mask = solvedPairMask(before);
			for (var p = 0; p < 4; p++) {
				if ((mask & (1 << p)) && !isPairSolved(cc, p)) {
					return false;
				}
			}
			return true;
		}
		if (hint.kind == 'oll') {
			return isOriented(cc);
		}
		return cc.isEqual(mathlib.CubieCube.SOLVED);
	}

	return {
		getStage: function(facelet) {
			return getStage(cubieFromFacelet(facelet));
		},
		solveNextPair: solveNextPair,
		solveCross: solveCross,
		solvePairs: solvePairs,
		solveLastLayer: solveLastLayer,
		getHint: getHint,
		warmUp: warmUp,
		buildAlllLut: buildAlllLut,
		ensureAlllLut: ensureAlllLut,
		alllReady: alllReady,
		alllLutSize: alllLutSize,
		buildOllLut: buildOllLut,
		buildPllLut: buildPllLut,
		ensureOllLut: ensureOllLut,
		ensurePllLut: ensurePllLut,
		normalizeFacelet: normalizeFacelet,
		canonicalSig: canonicalSig,
		orientFacelet: orientFacelet,
		simplifySeq: simplifySeq,
		inferMove: inferMove,
		invertToken: invertToken,
		pickSmartOrientation: pickSmartOrientation,
		algStates: algStates,
		algSteps: algSteps,
		validateAlg: validateAlg,
		slotFacelets: slotFacelets,
		F2L_PAIRS: F2L_PAIRS
	};
}

var cubeCopilot = execBoth(makeCubeCopilot, makeCubeCopilot);
