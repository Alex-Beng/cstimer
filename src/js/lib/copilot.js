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
	function solveNextPair(facelet, options) {
		options = options || {};
		var maxDepth = options.maxDepth || 8;
		var deadline = options.timeLimit ? $.now() + options.timeLimit : 0;
		var cc = cubieFromFacelet(facelet);
		var solved = solvedPairMask(cc);
		if (!isCrossSolved(cc) || solved == 0xf) {
			return null;
		}
		// cheap lower bound: skip pairs that provably need more than maxDepth
		var best = null;
		var timedOut = false;
		for (var i = 0; i < 4; i++) {
			if (solved & (1 << i)) {
				continue;
			}
			if (lowerBound(cc, i, solved) > maxDepth) {
				continue;
			}
			var path = searchPair(cc, i, maxDepth, solved, deadline);
			if (!path && deadline && $.now() > deadline) {
				timedOut = true;
			}
			if (path && (best == null || path.length < best.moves.length)) {
				best = {
					pair: i,
					name: F2L_PAIRS[i].name,
					moves: path,
					alg: path.join(' '),
					slot: slotFacelets(i),
					corner: F2L_PAIRS[i].corner,
					edge: F2L_PAIRS[i].edge,
					optimal: !timedOut
				};
			}
		}
		if (best) {
			best.optimal = !timedOut;
		}
		return best;
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
		return ALLL_READY ? ALLL : buildAlllLut();
	}

	function alllReady() {
		return ALLL_READY;
	}

	function alllLutSize() {
		return ALLL ? Object.keys(ALLL).length : 0;
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

	function solveLastLayer(facelet) {
		var cc = cubieFromFacelet(facelet);
		if (!isF2LDone(cc)) {
			return null;
		}
		if (isLastLayerSolved(cc)) {
			return {name: 'skip', alg: '', moves: []};
		}
		var hit = ensureAlllLut()[llKey(cc)];
		if (!hit) {
			// A solved layer rotated by a U turn is not stored in the LUT (the
			// algs there solve a scrambled layer), so handle it directly.
			var auf = solveByAuf(cc);
			if (auf) {
				return {
					name: 'AUF',
					alg: auf,
					moves: [auf]
				};
			}
			return null;
		}
		return {
			name: hit.name,
			alg: hit.alg,
			moves: hit.alg.split(/\s+/).filter(function(t) {
				return !!t;
			})
		};
	}

	function getHint(facelet, options) {
		var cc = cubieFromFacelet(facelet);
		var stage = getStage(cc);
		if (stage == 'f2l') {
			var pair = solveNextPair(facelet, options);
			if (!pair) {
				return {stage: 'f2l', name: 'no-solution'};
			}
			return {
				stage: 'f2l',
				pair: pair.pair,
				name: pair.name,
				moves: pair.moves,
				alg: pair.moves.join(' '),
				slot: pair.slot,
				corner: pair.corner,
				edge: pair.edge
			};
		}
		if (stage == 'll') {
			var ll = solveLastLayer(facelet);
			if (!ll) {
				return {stage: 'll', name: 'no-alg'};
			}
			return {stage: 'll', name: ll.name, moves: ll.moves, alg: ll.alg};
		}
		return {stage: stage};
	}

	return {
		getStage: function(facelet) {
			return getStage(cubieFromFacelet(facelet));
		},
		solveNextPair: solveNextPair,
		solveLastLayer: solveLastLayer,
		getHint: getHint,
		warmUp: warmUp,
		buildAlllLut: buildAlllLut,
		ensureAlllLut: ensureAlllLut,
		alllReady: alllReady,
		alllLutSize: alllLutSize,
		slotFacelets: slotFacelets,
		F2L_PAIRS: F2L_PAIRS
	};
}

var cubeCopilot = execBoth(makeCubeCopilot, makeCubeCopilot);
