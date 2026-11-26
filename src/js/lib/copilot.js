"use strict";

/**
 * Cube Copilot: stage-aware hints for 3x3 solves.
 *
 * Pure logic (no DOM) shared by the main thread and the scramble worker.
 * Input is a facelet string in csTimer order ("URFDLB").  Hints:
 *   - CFOP: the corner/edge pair needing the fewest moves while keeping the
 *     cross and already solved pairs intact (no algorithm is shown)
 *   - CFOP last layer: OLL/PLL, COLL, ZBLL and ALLL (1LLL) recognition through
 *     lazily built lookup tables
 *   - Roux: CMLL algorithms once both blocks are complete
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

	/** first-layer corners: bit i is set when D corner 4+i is solved */
	function solvedCornerMask(cc) {
		var mask = 0;
		for (var i = 0; i < 4; i++) {
			if (cc.ca[4 + i] == 4 + i) {
				mask |= 1 << i;
			}
		}
		return mask;
	}

	function isFirstLayerDone(cc) {
		return isCrossSolved(cc) && solvedCornerMask(cc) == 0xf;
	}

	function cornersOriented(cc) {
		for (var i = 0; i < 4; i++) {
			if ((cc.ca[i] >> 3) != 0) {
				return false;
			}
		}
		return true;
	}

	/** orientation of the four U-layer edge pieces, wherever they sit */
	function llEdgesOriented(cc) {
		for (var p = 0; p < 4; p++) {
			for (var i = 0; i < 12; i++) {
				if ((cc.ea[i] >> 1) == p) {
					if (cc.ea[i] & 1) {
						return false;
					}
					break;
				}
			}
		}
		return true;
	}

	/** orientation of the four U-layer corner pieces, wherever they sit */
	function llCornersOriented(cc) {
		for (var p = 0; p < 4; p++) {
			for (var i = 0; i < 8; i++) {
				if ((cc.ca[i] & 7) == p) {
					if ((cc.ca[i] >> 3) != 0) {
						return false;
					}
					break;
				}
			}
		}
		return true;
	}

	/**
	 * corner permutation only, ignoring a whole-layer AUF: the four U corners
	 * must sit in a cyclic shift of their own slots, so the relative order of
	 * the corners is what matters
	 */
	function cornersPermuted(cc) {
		var shift = (cc.ca[0] & 7) % 4;
		for (var i = 0; i < 4; i++) {
			if ((cc.ca[i] & 7) != (i + shift) % 4) {
				return false;
			}
		}
		return true;
	}

	/** both Roux blocks (everything except the M slice and U layer) */
	function isF2BDone(cc) {
		for (var c = 4; c < 8; c++) {
			if (cc.ca[c] != c) {
				return false;
			}
		}
		for (var e = 4; e < 12; e++) {
			if (e == 5 || e == 7) {
				continue;
			}
			if (cc.ea[e] != e * 2) {
				return false;
			}
		}
		return true;
	}

	function isLastLayerSolved(cc) {
		return cc.isEqual(mathlib.CubieCube.SOLVED);
	}

	/** 'cross' | 'f2l' | 'll' | 'solved' (CFOP) */
	function getStage(cc, method) {
		if (method == 'roux') {
			return getRouxStage(cc);
		}
		if (method == 'lbl') {
			return getLblStage(cc);
		}
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

	/** 'blocks' | 'cmll' | 'lse' | 'solved' (Roux) */
	function getRouxStage(cc) {
		if (isLastLayerSolved(cc)) {
			return 'solved';
		}
		if (!isF2BDone(cc)) {
			return 'blocks';
		}
		return cornersSolvedOnTop(cc) ? 'lse' : 'cmll';
	}

	/**
	 * 'cross' | 'fl' | 'sl' | 'eoll' | 'ocll' | 'cpll' | 'epll' | 'solved'
	 * (layer by layer: first-layer corners, second-layer edges, then the four
	 * beginner top-layer steps)
	 */
	function getLblStage(cc) {
		if (isLastLayerSolved(cc)) {
			return 'solved';
		}
		if (!isCrossSolved(cc)) {
			return 'cross';
		}
		if (solvedCornerMask(cc) != 0xf) {
			return 'fl';
		}
		if (!isF2LDone(cc)) {
			return 'sl';
		}
		if (!edgesOriented(cc)) {
			return 'eoll';
		}
		if (!cornersOriented(cc)) {
			return 'ocll';
		}
		if (!cornersSolvedOnTop(cc)) {
			return 'cpll';
		}
		return 'epll';
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
		var cubes = getMoveCubes();
		// transition of one (position, orientation) pair per move; this lets
		// the BFS work on the 331776 integer keys instead of full cubes,
		// which is ~25x faster than multiplying CubieCubes
		var trans = new Uint8Array(24 * 18);
		for (var pv = 0; pv < 24; pv++) {
			var pos = ~~(pv / 2);
			var ori = pv & 1;
			for (var m = 0; m < 18; m++) {
				var cube = cubes[m];
				for (var ed = 0; ed < 12; ed++) {
					if ((cube.ea[ed] >> 1) == pos) {
						trans[pv * 18 + m] = (ed << 1) | (ori ^ (cube.ea[ed] & 1));
						break;
					}
				}
			}
		}
		var table = new Int8Array(CROSS_STATES);
		table.fill(-1);
		var start = crossKey(new mathlib.CubieCube());
		table[start] = 0;
		var frontier = [start];
		var depth = 0;
		while (frontier.length && depth < 16) {
			depth++;
			var next = [];
			for (var f = 0; f < frontier.length; f++) {
				var key = frontier[f];
				var v3 = key % 24;
				key = (key - v3) / 24;
				var v2 = key % 24;
				key = (key - v2) / 24;
				var v1 = key % 24;
				var v0 = (key - v1) / 24;
				for (var m2 = 0; m2 < 18; m2++) {
					var nk = (((trans[v0 * 18 + m2] * 24 + trans[v1 * 18 + m2]) * 24 +
						trans[v2 * 18 + m2]) * 24 + trans[v3 * 18 + m2]);
					if (table[nk] >= 0) {
						continue;
					}
					table[nk] = depth;
					next.push(nk);
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

	// ------------------------------------------------- piece pruning (LBL)

	var PIECE_PRUN = {corner: {}, edge: {}};

	/** slot/orientation coordinate of a single corner or edge (0..23) */
	function pieceCoord(cc, kind, piece) {
		var arr = kind == 'corner' ? cc.ca : cc.ea;
		var n = kind == 'corner' ? 8 : 12;
		for (var p = 0; p < n; p++) {
			var perm = kind == 'corner' ? (arr[p] & 7) : (arr[p] >> 1);
			if (perm == piece) {
				return kind == 'corner' ? p * 3 + (arr[p] >> 3) : p * 2 + (arr[p] & 1);
			}
		}
		return -1;
	}

	/** distance of one piece back to its home slot, all other pieces ignored */
	function piecePrun(kind, piece) {
		var cache = PIECE_PRUN[kind];
		if (cache[piece]) {
			return cache[piece];
		}
		var start = new mathlib.CubieCube();
		var startKey = pieceCoord(start, kind, piece);
		var dist = {};
		var reps = {};
		var queue = [startKey];
		var moves = getMoveCubes();
		dist[startKey] = 0;
		reps[startKey] = start;
		for (var qi = 0; qi < queue.length; qi++) {
			var key = queue[qi];
			var cc = reps[key];
			for (var m = 0; m < moves.length; m++) {
				var ns = mul(cc, moves[m]);
				var nkey = pieceCoord(ns, kind, piece);
				if (nkey < 0 || nkey in dist) {
					continue;
				}
				dist[nkey] = dist[key] + 1;
				reps[nkey] = ns;
				queue.push(nkey);
			}
		}
		cache[piece] = dist;
		return dist;
	}

	function pieceDistance(cc, kind, piece) {
		var v = piecePrun(kind, piece)[pieceCoord(cc, kind, piece)];
		return v == null ? 99 : v;
	}

	/** number of solved F2L pairs (0..4) */
	function pairCountOf(facelet) {
		var mask = solvedPairMask(cubieFromFacelet(facelet));
		var n = 0;
		for (var i = 0; i < 4; i++) {
			if (mask & (1 << i)) {
				n++;
			}
		}
		return n;
	}

	/** bitmask of solved F2L pairs, bit i = pair i */
	function pairMask(facelet) {
		return solvedPairMask(cubieFromFacelet(facelet));
	}

	/**
	 * Where the pair's corner and edge currently are, plus the pruning
	 * distance (the minimum number of moves needed to insert the pair).
	 * No search is performed: this is the live F2L hint.
	 */
	function pairInfoOf(cc, pairIdx) {
		var pair = F2L_PAIRS[pairIdx];
		var cornerAt = -1;
		for (var p = 0; p < 8; p++) {
			if ((cc.ca[p] & 7) == pair.corner) {
				cornerAt = p;
				break;
			}
		}
		var edgeAt = -1;
		for (var q = 0; q < 12; q++) {
			if ((cc.ea[q] >> 1) == pair.edge) {
				edgeAt = q;
				break;
			}
		}
		var pieces = [];
		if (cornerAt >= 0) {
			pieces = pieces.concat(mathlib.CubieCube.cFacelet[cornerAt]);
		}
		if (edgeAt >= 0) {
			pieces = pieces.concat(mathlib.CubieCube.eFacelet[edgeAt]);
		}
		return {
			pair: pairIdx,
			name: pair.name,
			corner: pair.corner,
			edge: pair.edge,
			cornerAt: cornerAt,
			edgeAt: edgeAt,
			slot: slotFacelets(pairIdx),
			pieces: pieces,
			distance: pairDistance(pairIdx, cc),
			solved: isPairSolved(cc, pairIdx)
		};
	}

	/** sticker colours of a corner/edge, independent of the cube orientation */
	function stickerKey(facelet, facelets) {
		var ret = [];
		for (var i = 0; i < facelets.length; i++) {
			ret.push(facelet[facelets[i]]);
		}
		return ret.sort().join('');
	}

	function annotateColors(facelet, info) {
		if (info && info.cornerAt >= 0 && info.edgeAt >= 0) {
			info.cornerColors = stickerKey(facelet, mathlib.CubieCube.cFacelet[info.cornerAt]);
			info.edgeColors = stickerKey(facelet, mathlib.CubieCube.eFacelet[info.edgeAt]);
		}
		return info;
	}

	function pairInfo(facelet, pairIdx) {
		return annotateColors(facelet, pairInfoOf(cubieFromFacelet(facelet), pairIdx));
	}

	function faceOfFacelet(idx) {
		return ~~(idx / 9);
	}

	/**
	 * A "basic" (standard) F2L case: the pair is already joined in the U
	 * layer, so a three move trigger (R U R' and the like) inserts it.
	 * Verified on random positions: every joined pair in the U layer can be
	 * inserted in exactly three moves, while split pairs cannot.
	 */
	function basicPair(facelet, pairIdx) {
		var pair = F2L_PAIRS[pairIdx];
		if (!pair) {
			return false;
		}
		var cc = cubieFromFacelet(facelet);
		var cornerAt = -1;
		for (var p = 0; p < 8; p++) {
			if ((cc.ca[p] & 7) == pair.corner) {
				cornerAt = p;
				break;
			}
		}
		var edgeAt = -1;
		for (var q = 0; q < 12; q++) {
			if ((cc.ea[q] >> 1) == pair.edge) {
				edgeAt = q;
				break;
			}
		}
		// both pieces in the U layer
		if (cornerAt < 0 || cornerAt > 3 || edgeAt < 0 || edgeAt > 3) {
			return false;
		}
		var cf = mathlib.CubieCube.cFacelet[cornerAt];
		var ef = mathlib.CubieCube.eFacelet[edgeAt];
		// joined: the corner and the edge share a matching side sticker
		for (var i = 1; i < 3; i++) {
			if (faceOfFacelet(cf[i]) == faceOfFacelet(ef[1]) &&
					facelet[cf[i]] == facelet[ef[1]]) {
				return true;
			}
		}
		return false;
	}

	function findPieceAt(facelet, facelets, key) {
		for (var p = 0; p < facelets.length; p++) {
			if (stickerKey(facelet, facelets[p]) == key) {
				return p;
			}
		}
		return -1;
	}

	function pieceMatchesCenters(facelet, facelets) {
		for (var i = 0; i < facelets.length; i++) {
			var idx = facelets[i];
			if (facelet[idx] != facelet[(~~(idx / 9)) * 9 + 4]) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Find the same physical corner/edge again from their colour keys.  A
	 * whole-cube rotation moves the stickers but keeps their colours, so the
	 * hint stays on the same pair instead of the same slot.
	 */
	function locatePair(facelet, cornerColors, edgeColors) {
		if (!facelet || !cornerColors || !edgeColors) {
			return null;
		}
		var cFacelet = mathlib.CubieCube.cFacelet;
		var eFacelet = mathlib.CubieCube.eFacelet;
		var cornerAt = findPieceAt(facelet, cFacelet, cornerColors);
		var edgeAt = findPieceAt(facelet, eFacelet, edgeColors);
		if (cornerAt < 0 || edgeAt < 0) {
			return null;
		}
		return {
			cornerAt: cornerAt,
			edgeAt: edgeAt,
			cornerColors: cornerColors,
			edgeColors: edgeColors,
			pieces: cFacelet[cornerAt].concat(eFacelet[edgeAt]),
			solved: pieceMatchesCenters(facelet, cFacelet[cornerAt]) &&
				pieceMatchesCenters(facelet, eFacelet[edgeAt])
		};
	}

	/** cheapest pair by the pruning distance only (admissible but not exact) */
	function approxMinPair(facelet) {
		var cc = cubieFromFacelet(facelet);
		if (!isCrossSolved(cc)) {
			return null;
		}
		var best = null;
		for (var i = 0; i < 4; i++) {
			var info = pairInfoOf(cc, i);
			if (info.solved) {
				continue;
			}
			if (!best || info.distance < best.distance) {
				best = info;
			}
		}
		best = annotateColors(facelet, best);
		if (best) {
			best.exact = false;
		}
		return best;
	}

	/**
	 * Admissible lower bound for inserting `pairIdx` without breaking the
	 * cross or any protected pair, so it can prune the exact search.
	 */
	function pairLowerBound(cc, pairIdx, protectMask) {
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

	function pairGoal(cc, pairIdx, protectMask) {
		if (!isPairSolved(cc, pairIdx) || !isCrossSolved(cc)) {
			return false;
		}
		for (var p = 0; p < 4; p++) {
			if ((protectMask & (1 << p)) && !isPairSolved(cc, p)) {
				return false;
			}
		}
		return true;
	}

	function searchPairDepth(cc, pairIdx, depth, protectMask, deadline) {
		var moves = getMoveCubes();
		var path = [];

		function dfs(state, remaining, lastFace) {
			if (deadline && $.now() > deadline) {
				return null;
			}
			if (pairLowerBound(state, pairIdx, protectMask) > remaining) {
				return null;
			}
			if (remaining == 0) {
				return pairGoal(state, pairIdx, protectMask) ? path.slice() : null;
			}
			for (var m = 0; m < moves.length; m++) {
				var face = ~~(m / 3);
				if (face == lastFace) {
					continue;
				}
				var ns = mul(state, moves[m]);
				path.push(MOVE_NAMES[m]);
				var found = dfs(ns, remaining - 1, face);
				if (found) {
					return found;
				}
				path.pop();
			}
			return null;
		}
		return dfs(cc, depth, -1);
	}

	/**
	 * The unsolved pair that really needs the fewest moves to be inserted,
	 * keeping the cross and the already solved pairs intact.  Depths are
	 * tried in increasing order and the search stops at the first depth where
	 * any pair fits, so the usual cost is only a few milliseconds.  If the
	 * time limit is hit, fall back to the pruning-distance ranking.  The last
	 * remaining pair is returned directly, without searching.
	 */
	function minPair(facelet, options) {
		options = options || {};
		var cc = cubieFromFacelet(facelet);
		var solved = solvedPairMask(cc);
		if (!isCrossSolved(cc) || solved == 0xf) {
			return null;
		}
		var maxDepth = options.maxDepth || 11;
		var deadline = $.now() + (options.timeLimit || 1000);
		var unsolved = [];
		for (var u = 0; u < 4; u++) {
			if (!(solved & (1 << u))) {
				unsolved.push(u);
			}
		}
		// only one pair left: it is the hint, no need to search for it
		if (unsolved.length == 1) {
			var only = annotateColors(facelet, pairInfoOf(cc, unsolved[0]));
			only.exact = false;
			return only;
		}
		var cand = [];
		var minLb = 99;
		for (var i = 0; i < unsolved.length; i++) {
			var lb = pairLowerBound(cc, unsolved[i], solved);
			cand.push({i: unsolved[i], lb: lb, found: null});
			if (lb < minLb) {
				minLb = lb;
			}
		}
		for (var depth = minLb; depth <= maxDepth; depth++) {
			for (var c = 0; c < cand.length; c++) {
				var item = cand[c];
				if (item.lb > depth || item.found) {
					continue;
				}
				var path = searchPairDepth(cc, item.i, depth, solved, deadline);
				if (path) {
					item.found = path;
					item.depth = depth;
				}
				if ($.now() > deadline) {
					break;
				}
			}
			for (var c2 = 0; c2 < cand.length; c2++) {
				if (cand[c2].found) {
					var info = annotateColors(facelet, pairInfoOf(cc, cand[c2].i));
					info.distance = cand[c2].depth;
					info.moves = cand[c2].found;
					info.exact = true;
					return info;
				}
			}
			if ($.now() > deadline) {
				break;
			}
		}
		return approxMinPair(facelet);
	}

	// ------------------------------------------------------------ LBL search

	function middleEdgeMask(cc) {
		var mask = 0;
		for (var i = 0; i < 4; i++) {
			if (cc.ea[8 + i] == (8 + i) * 2) {
				mask |= 1 << i;
			}
		}
		return mask;
	}

	function lblLowerBound(cc, kind, target, protect) {
		var h = crossDistance(cc);
		function bump(d) {
			if (d < 99 && d > h) {
				h = d;
			}
		}
		if (kind == 'corner') {
			bump(pieceDistance(cc, 'corner', 4 + target));
			for (var i = 0; i < 4; i++) {
				if (protect & (1 << i)) {
					bump(pieceDistance(cc, 'corner', 4 + i));
				}
			}
		} else {
			for (var c = 0; c < 4; c++) {
				bump(pieceDistance(cc, 'corner', 4 + c));
			}
			// the D corner and the middle edge of the target slot must both be
			// home, so the F2L pair distance is an admissible lower bound
			bump(pairDistance(target, cc));
			for (var e = 0; e < 4; e++) {
				if (protect & (1 << e)) {
					bump(pieceDistance(cc, 'edge', 8 + e));
				}
			}
		}
		return h;
	}

	function lblGoal(cc, kind, target, protect) {
		if (!isCrossSolved(cc)) {
			return false;
		}
		if (kind == 'corner') {
			if (cc.ca[4 + target] != 4 + target) {
				return false;
			}
			for (var i = 0; i < 4; i++) {
				if (protect & (1 << i)) {
					if (cc.ca[4 + i] != 4 + i) {
						return false;
					}
				}
			}
			return true;
		}
		for (var c = 0; c < 4; c++) {
			if (cc.ca[4 + c] != 4 + c) {
				return false;
			}
		}
		if (cc.ea[8 + target] != (8 + target) * 2) {
			return false;
		}
		for (var e = 0; e < 4; e++) {
			if (protect & (1 << e)) {
				if (cc.ea[8 + e] != (8 + e) * 2) {
					return false;
				}
			}
		}
		return true;
	}

	function searchLblDepth(cc, kind, target, protect, depth, deadline) {
		var moves = getMoveCubes();
		var path = [];

		function dfs(state, remaining, lastFace) {
			if (deadline && $.now() > deadline) {
				return null;
			}
			if (lblLowerBound(state, kind, target, protect) > remaining) {
				return null;
			}
			if (remaining == 0) {
				return lblGoal(state, kind, target, protect) ? path.slice() : null;
			}
			for (var m = 0; m < moves.length; m++) {
				var face = ~~(m / 3);
				if (face == lastFace) {
					continue;
				}
				var ns = mul(state, moves[m]);
				path.push(MOVE_NAMES[m]);
				var found = dfs(ns, remaining - 1, face);
				if (found) {
					return found;
				}
				path.pop();
			}
			return null;
		}
		return dfs(cc, depth, -1);
	}

	function lblInfo(facelet, cc, kind, target, path, protect) {
		var piece = (kind == 'corner' ? 4 : 8) + target;
		var coord = pieceCoord(cc, kind, piece);
		var at = kind == 'corner' ? ~~(coord / 3) : ~~(coord / 2);
		var facelets = kind == 'corner' ? mathlib.CubieCube.cFacelet[at] : mathlib.CubieCube.eFacelet[at];
		return {
			kind: kind == 'corner' ? 'lblcorner' : 'lblmid',
			pair: target,
			name: F2L_PAIRS[target].name,
			key: 'LBL/' + (kind == 'corner' ? 'CORNER/' : 'MID/') + target,
			base: path.slice(),
			pre: '',
			post: '',
			alg: path.join(' '),
			corner: kind == 'corner' ? piece : -1,
			edge: kind == 'edge' ? piece : -1,
			cornerAt: kind == 'corner' ? at : -1,
			edgeAt: kind == 'edge' ? at : -1,
			cornerColors: kind == 'corner' ? stickerKey(facelet, facelets) : null,
			edgeColors: kind == 'edge' ? stickerKey(facelet, facelets) : null,
			pieces: facelets.slice(),
			protect: protect,
			distance: path.length,
			moves: path,
			exact: true,
			solved: false
		};
	}

	/** follow a first/second-layer hint after the cube (and the piece) moved */
	function locateLbl(facelet, hint) {
		if (!facelet || !hint) {
			return null;
		}
		var isCorner = hint.kind == 'lblcorner';
		var facelets = isCorner ? mathlib.CubieCube.cFacelet : mathlib.CubieCube.eFacelet;
		var key = isCorner ? hint.cornerColors : hint.edgeColors;
		if (!key) {
			return null;
		}
		var at = findPieceAt(facelet, facelets, key);
		if (at < 0) {
			return null;
		}
		return {
			at: at,
			pieces: facelets[at].slice(),
			solved: pieceMatchesCenters(facelet, facelets[at])
		};
	}

	function lblSearch(facelet, kind, options) {
		options = options || {};
		var cc = cubieFromFacelet(facelet);
		if (!isCrossSolved(cc)) {
			return null;
		}
		if (kind == 'edge' && solvedCornerMask(cc) != 0xf) {
			return null;
		}
		var solved = kind == 'corner' ? solvedCornerMask(cc) : middleEdgeMask(cc);
		if (solved == 0xf) {
			return null;
		}
		var protect = solved;
		var cand = [];
		var minLb = 99;
		for (var i = 0; i < 4; i++) {
			if (solved & (1 << i)) {
				continue;
			}
			var lb = lblLowerBound(cc, kind, i, protect);
			cand.push({i: i, lb: lb});
			if (lb < minLb) {
				minLb = lb;
			}
		}
		var maxDepth = options.maxDepth || 11;
		var deadline = $.now() + (options.timeLimit || 1000);
		for (var depth = minLb; depth <= maxDepth; depth++) {
			for (var c = 0; c < cand.length; c++) {
				var item = cand[c];
				if (item.lb > depth) {
					continue;
				}
				var path = searchLblDepth(cc, kind, item.i, protect, depth, deadline);
				if (path) {
					return lblInfo(facelet, cc, kind, item.i, path, protect);
				}
				if ($.now() > deadline) {
					return null;
				}
			}
		}
		return null;
	}

	/** shortest first-layer corner insertion (cross and solved corners kept) */
	function lblCornerHint(facelet, options) {
		return lblSearch(facelet, 'corner', options);
	}

	/**
	 * Second-layer edge insertions, one short algorithm per slot and per
	 * position/orientation of the slot's edge.  The first layer is solved and
	 * the edge sits in the U layer or in one of the other middle slots, so the
	 * case depends only on that coordinate; using a table keeps the hint instant
	 * (an exact search needs seconds for the harder cases).
	 */
	var LBL_SL_DATA = [
		{
			0: "R' U2 B' R' B U2 R", 1: "R U2 R' U2 F' U' F",
			2: "F' U2 F U2 R U R'", 3: "R' U2 B' R B U2 R",
			4: "R U2 B U2 B' U2 R'", 5: "R U R' U' F' U' F",
			6: "R U2 B U B' U2 R'", 7: "F U2 L F2 L' U2 F'",
			17: "R U' R U2 F R2 F' U2 R2", 18: "F2 U2 F2 U2 F2",
			19: "F U L F' L' F'", 20: "R2 U D B2 U' D' R2",
			21: "R2 D B' R' U R D' R2", 22: "R2 U2 R2 U2 R2",
			23: "R F R F' U' R'"
		},
		{
			0: "R U R2 D' F2 D R", 1: "F U2 R U R' U2 F'",
			2: "F U2 R U' R' U2 F'", 3: "R' F2 U' F' U F2 R",
			4: "R B2 D L D' B2 R'", 5: "R' F2 U' F U F2 R",
			6: "R B2 D L' D' B2 R'", 7: "F U2 R U2 R' U2 F'",
			16: "F2 U2 F2 U2 F2", 17: "F L F L' U' F'",
			19: "F U' F U2 L F2 L' U2 F2", 20: "R2 D2 L2 D2 R2",
			21: "L U B L' B' L'", 22: "R F2 U R U' F2 R'",
			23: "R2 D' R U R' F' D R2"
		},
		{
			0: "R D B2 D' R2 U R", 1: "F R2 D B' D' R2 F'",
			2: "R' F2 D' L D F2 R", 3: "F U F2 D' L2 D F",
			4: "R' F2 D' L' D F2 R", 5: "R B2 U B' U' B2 R'",
			6: "F' L2 U' L U L2 F", 7: "R B2 U B U' B2 R'",
			16: "R2 U D B2 U' D' R2", 17: "R2 D R' U' R B D' R2",
			18: "R2 D2 L2 D2 R2", 19: "L B L B' U' L'",
			21: "L U' L U2 B L2 B' U2 L2", 22: "F2 D2 B2 D2 F2",
			23: "B U R B' R' B'"
		},
		{
			0: "R U2 F R F' U2 R'", 1: "R' U2 R U2 B U B'",
			2: "R B' R' B R' U' R", 3: "F R2 U R2 U' R2 F'",
			4: "R U2 F R2 F' U2 R'", 5: "R' U R U B U' B'",
			6: "F R2 U R' U' R2 F'", 7: "R U2 F R' F' U2 R'",
			16: "R2 U2 R2 U2 R2", 17: "R U F R' F' R'",
			18: "R F2 U R' U' F2 R'", 19: "R2 D' F R U' R' D R2",
			20: "F2 D2 B2 D2 F2", 21: "B R B R' U' B'",
			23: "R2 U2 B' R2 B U2 R U' R"
		}
	];

	/** position/orientation of one edge piece: pos*2 + ori (0..23) */
	function edgeCoord(cc, piece) {
		for (var p = 0; p < 12; p++) {
			if ((cc.ea[p] >> 1) == piece) {
				return p * 2 + (cc.ea[p] & 1);
			}
		}
		return -1;
	}

	/** precomputed second-layer insertion (first layer and solved edges kept) */
	function lblMidHint(facelet, options) {
		options = options || {};
		var cc = cubieFromFacelet(facelet);
		if (!isCrossSolved(cc) || solvedCornerMask(cc) != 0xf) {
			return null;
		}
		var solved = middleEdgeMask(cc);
		if (solved == 0xf) {
			return null;
		}
		var protect = solved;
		for (var i = 0; i < 4; i++) {
			if (solved & (1 << i)) {
				continue;
			}
			var coord = edgeCoord(cc, 8 + i);
			var alg = coord < 0 ? null : LBL_SL_DATA[i][coord];
			if (!alg) {
				continue;
			}
			var tokens = tokensOf(alg);
			var info = lblInfo(facelet, cc, 'edge', i, tokens, protect);
			// the table assumes the maximal set of intact edges; a state whose
			// protected edges differ falls back to the search below
			if (validateAlg(facelet, tokens, info)) {
				return info;
			}
		}
		return lblSearch(facelet, 'edge', options);
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

	/** the U-layer edges are flipped correctly (ZBLL/COLL precondition) */
	function edgesOriented(cc) {
		for (var i = 0; i < 4; i++) {
			if (cc.ea[i] & 1) {
				return false;
			}
		}
		return true;
	}

	function cornersSolvedOnTop(cc) {
		for (var i = 0; i < 4; i++) {
			if (cc.ca[i] != i) {
				return false;
			}
		}
		return true;
	}

	/** corner permutation/orientation only: COLL recognition ignores the edges */
	function cornerKey(cc) {
		var ret = [];
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ca[i] & 7, cc.ca[i] >> 3);
		}
		return ret.join(',');
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

	// ------------------------------------------------------------ ZBLL / COLL

	var ZBLL = null;
	var ZBLL_READY = false;
	var COLL_LUT = null;

	function buildZbllLut() {
		ZBLL = {};
		var list = (typeof ZBLL_DATA == 'undefined') ? [] : ZBLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var state = inverseState(seq);
					if (!isF2LDone(state) || !edgesOriented(state)) {
						continue;
					}
					var test = cloneCube(state);
					for (var k = 0; k < seq.length; k++) {
						test.selfMoveStr(seq[k]);
					}
					if (!isLastLayerSolved(test)) {
						continue;
					}
					var key = llKey(state);
					if (!isLegalKey(key)) {
						continue;
					}
					var disp = seq.join(' ');
					if (!(key in ZBLL) || disp.length < ZBLL[key].alg.length) {
						ZBLL[key] = {name: e.name, alg: disp, len: disp.split(/\s+/).length};
					}
				}
			}
		}
		ZBLL_READY = true;
		return ZBLL;
	}

	function ensureZbllLut() {
		return ZBLL || buildZbllLut();
	}

	function zbllReady() {
		return ZBLL_READY;
	}

	function zbllLutSize() {
		return ZBLL ? Object.keys(ZBLL).length : 0;
	}

	function zbllHint(cc) {
		var hit = ensureZbllLut()[llKey(cc)];
		if (!hit) {
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'zbll',
			name: hit.name,
			key: 'ZBLL/' + hit.name,
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	function buildCollLut() {
		COLL_LUT = {};
		var list = (typeof COLL_DATA == 'undefined') ? [] : COLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var state = inverseState(seq);
					if (!isF2LDone(state) || !edgesOriented(state)) {
						continue;
					}
					var test = cloneCube(state);
					for (var k = 0; k < seq.length; k++) {
						test.selfMoveStr(seq[k]);
					}
					if (!isF2LDone(test) || !edgesOriented(test) || !cornersSolvedOnTop(test)) {
						continue;
					}
					var key = cornerKey(state);
					var disp = seq.join(' ');
					if (!(key in COLL_LUT) || disp.length < COLL_LUT[key].alg.length) {
						COLL_LUT[key] = {name: e.name, alg: disp, len: disp.split(/\s+/).length};
					}
				}
			}
		}
		return COLL_LUT;
	}

	function ensureCollLut() {
		return COLL_LUT || buildCollLut();
	}

	function collHint(cc) {
		var hit = ensureCollLut()[cornerKey(cc)];
		if (!hit) {
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'coll',
			name: hit.name,
			key: 'COLL/' + hit.name,
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	// ------------------------------------------------------------ Roux CMLL

	var CMLL = null;

	function buildCmllLut() {
		CMLL = {};
		var list = (typeof CMLL_DATA == 'undefined') ? [] : CMLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var state = inverseState(seq);
					if (!isF2BDone(state)) {
						continue;
					}
					var test = cloneCube(state);
					for (var k = 0; k < seq.length; k++) {
						test.selfMoveStr(seq[k]);
					}
					if (!isF2BDone(test) || !cornersSolvedOnTop(test)) {
						continue;
					}
					var key = cornerKey(state);
					var disp = seq.join(' ');
					if (!(key in CMLL) || disp.length < CMLL[key].alg.length) {
						CMLL[key] = {name: e.name, alg: disp, len: disp.split(/\s+/).length};
					}
				}
			}
		}
		return CMLL;
	}

	function ensureCmllLut() {
		return CMLL || buildCmllLut();
	}

	function cmllLutSize() {
		return CMLL ? Object.keys(CMLL).length : 0;
	}

	function cmllHint(cc) {
		var hit = ensureCmllLut()[cornerKey(cc)];
		if (!hit) {
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'cmll',
			name: hit.name,
			key: 'CMLL/' + hit.name,
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	function solveCMLL(facelet) {
		var cc = cubieFromFacelet(facelet);
		if (!isF2BDone(cc) || cornersSolvedOnTop(cc)) {
			return null;
		}
		var hit = cmllHint(cc);
		if (hit) {
			return hit;
		}
		// corners are solved up to an AUF, so only the top layer needs aligning
		for (var k = 1; k < 4; k++) {
			var test = cloneCube(cc);
			for (var i = 0; i < k; i++) {
				test.selfMoveStr('U');
			}
			if (cornersSolvedOnTop(test)) {
				var alg = AUF_STR[k];
				return {
					kind: 'cmll',
					name: 'AUF',
					key: 'CMLL/AUF',
					base: [alg],
					pre: '',
					post: '',
					moves: [alg],
					alg: alg
				};
			}
		}
		return null;
	}

	// ------------------------------------------------------ LBL top layer

	var LBL_EO = null;
	var LBL_OCLL = null;
	var LBL_CPLL = null;

	function eoKey(cc) {
		var ret = 0;
		for (var i = 0; i < 4; i++) {
			ret |= (cc.ea[i] & 1) << i;
		}
		return ret;
	}

	function cornerOriKey(cc) {
		var ret = [];
		for (var i = 0; i < 4; i++) {
			ret.push(cc.ca[i] >> 3);
		}
		return ret.join(',');
	}

	function buildEollLut() {
		LBL_EO = {};
		var t1 = 'F R U R\' U\' F\''.split(' ');
		var t2 = 'F U R U\' R\' F\''.split(' ');
		var seqs = [];
		function add(seq) {
			seqs.push(seq);
		}
		for (var a = 0; a < 4; a++) {
			add(tokensOf(AUF_STR[a]).concat(t1));
			add(tokensOf(AUF_STR[a]).concat(t2));
		}
		for (var pre = 0; pre < 4; pre++) {
			for (var mid = 0; mid < 4; mid++) {
				for (var post = 0; post < 4; post++) {
					add(tokensOf(AUF_STR[pre]).concat(t1, tokensOf(AUF_STR[mid]), t2, tokensOf(AUF_STR[post])));
					add(tokensOf(AUF_STR[pre]).concat(t2, tokensOf(AUF_STR[mid]), t1, tokensOf(AUF_STR[post])));
				}
			}
		}
		for (var i = 0; i < seqs.length; i++) {
			var seq = seqs[i];
			var state = inverseState(seq);
			if (!isF2LDone(state) || edgesOriented(state)) {
				continue;
			}
			var test = cloneCube(state);
			for (var k = 0; k < seq.length; k++) {
				test.selfMoveStr(seq[k]);
			}
			if (!isF2LDone(test) || !edgesOriented(test)) {
				continue;
			}
			var key = eoKey(state);
			var disp = seq.join(' ');
			if (!(key in LBL_EO) || disp.length < LBL_EO[key].alg.length) {
				LBL_EO[key] = {name: 'EO', alg: disp, len: disp.split(/\s+/).length};
			}
		}
		return LBL_EO;
	}

	function ensureEollLut() {
		return LBL_EO || buildEollLut();
	}

	function buildOcllLut() {
		LBL_OCLL = {};
		var list = (typeof OLL_DATA == 'undefined') ? [] : OLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var state = inverseState(seq);
					if (!isF2LDone(state) || !edgesOriented(state) || cornersOriented(state)) {
						continue;
					}
					var test = cloneCube(state);
					for (var k = 0; k < seq.length; k++) {
						test.selfMoveStr(seq[k]);
					}
					if (!isF2LDone(test) || !edgesOriented(test) || !cornersOriented(test)) {
						continue;
					}
					var key = cornerOriKey(state);
					var disp = seq.join(' ');
					if (!(key in LBL_OCLL) || disp.length < LBL_OCLL[key].alg.length) {
						LBL_OCLL[key] = {name: e.name, alg: disp, len: disp.split(/\s+/).length};
					}
				}
			}
		}
		return LBL_OCLL;
	}

	function ensureOcllLut() {
		return LBL_OCLL || buildOcllLut();
	}

	function buildCpllLut() {
		LBL_CPLL = {};
		var list = (typeof CMLL_DATA == 'undefined') ? [] : CMLL_DATA;
		for (var i = 0; i < list.length; i++) {
			var e = parseEntry(list[i]);
			if (!e) {
				continue;
			}
			for (var pre = 0; pre < 4; pre++) {
				for (var post = 0; post < 4; post++) {
					var seq = tokensOf(AUF_STR[pre]).concat(e.tokens, tokensOf(AUF_STR[post]));
					var state = inverseState(seq);
					if (!isF2LDone(state)) {
						continue;
					}
					var test = cloneCube(state);
					for (var k = 0; k < seq.length; k++) {
						test.selfMoveStr(seq[k]);
					}
					if (!isF2LDone(test) || !cornersSolvedOnTop(test) || !edgesOriented(test)) {
						continue;
					}
					var key = cornerKey(state);
					var disp = seq.join(' ');
					if (!(key in LBL_CPLL) || disp.length < LBL_CPLL[key].alg.length) {
						LBL_CPLL[key] = {name: e.name, alg: disp, len: disp.split(/\s+/).length};
					}
				}
			}
		}
		return LBL_CPLL;
	}

	function ensureCpllLut() {
		return LBL_CPLL || buildCpllLut();
	}

	function eollHint(cc) {
		if (edgesOriented(cc)) {
			return null;
		}
		var hit = ensureEollLut()[eoKey(cc)];
		if (!hit) {
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'eoll',
			name: 'EO',
			key: 'LBL/EO/' + eoKey(cc),
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	function ocllHint(cc) {
		if (cornersOriented(cc)) {
			return null;
		}
		var hit = ensureOcllLut()[cornerOriKey(cc)];
		if (!hit) {
			return null;
		}
		var tokens = tokensOf(hit.alg);
		return {
			kind: 'ocll',
			name: hit.name,
			key: 'LBL/OCLL/' + hit.name,
			base: tokens.slice(),
			pre: '',
			post: '',
			moves: tokens,
			alg: hit.alg
		};
	}

	function cpllHint(cc) {
		if (!cornersOriented(cc) || cornersSolvedOnTop(cc)) {
			return null;
		}
		var hit = ensureCpllLut()[cornerKey(cc)];
		if (hit) {
			var tokens = tokensOf(hit.alg);
			return {
				kind: 'cpll',
				name: hit.name,
				key: 'LBL/CPLL/' + hit.name,
				base: tokens.slice(),
				pre: '',
				post: '',
				moves: tokens,
				alg: hit.alg
			};
		}
		// corners are solved up to an AUF
		for (var k = 1; k < 4; k++) {
			var test = cloneCube(cc);
			for (var i = 0; i < k; i++) {
				test.selfMoveStr('U');
			}
			if (cornersSolvedOnTop(test)) {
				var alg = AUF_STR[k];
				return {
					kind: 'cpll',
					name: 'AUF',
					key: 'LBL/CPLL/AUF',
					base: [alg],
					pre: '',
					post: '',
					moves: [alg],
					alg: alg
				};
			}
		}
		return null;
	}

	function epllHint(cc) {
		if (!cornersSolvedOnTop(cc) || isLastLayerSolved(cc)) {
			return null;
		}
		var hit = pllHint(cc);
		if (!hit) {
			return null;
		}
		return {
			kind: 'epll',
			name: hit.name,
			key: 'LBL/EPLL/' + hit.name,
			base: hit.base,
			pre: hit.pre,
			post: hit.post,
			moves: hit.moves,
			alg: hit.alg
		};
	}

	function lblKind(stage) {
		if (stage == 'fl') {
			return 'lblcorner';
		}
		if (stage == 'sl') {
			return 'lblmid';
		}
		return stage;
	}

	function solveLbl(facelet, stage, options) {
		if (stage == 'fl') {
			return lblCornerHint(facelet, options);
		}
		if (stage == 'sl') {
			return lblMidHint(facelet, options);
		}
		var cc = cubieFromFacelet(facelet);
		if (stage == 'eoll') {
			return eollHint(cc);
		}
		if (stage == 'ocll') {
			return ocllHint(cc);
		}
		if (stage == 'cpll') {
			return cpllHint(cc);
		}
		if (stage == 'epll') {
			return epllHint(cc);
		}
		return null;
	}

	function twoLookHint(cc) {
		if (!isOriented(cc)) {
			return ollHint(cc);
		}
		return pllHint(cc) || aufHint(cc);
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
		if (mode == 'zbll') {
			if (edgesOriented(cc)) {
				var z = zbllHint(cc);
				if (z) {
					return z;
				}
			}
			// no ZBLL case (edges unoriented or unknown): standard two-look
			var z2 = twoLookHint(cc);
			if (z2) {
				return z2;
			}
		}
		if (mode == 'coll') {
			if (edgesOriented(cc)) {
				if (cornersSolvedOnTop(cc)) {
					var ep = pllHint(cc) || aufHint(cc);
					if (ep) {
						return ep;
					}
				} else {
					var c = collHint(cc);
					if (c) {
						return c;
					}
				}
			}
			var c2 = twoLookHint(cc);
			if (c2) {
				return c2;
			}
		}
		if (mode == 'op') {
			var ll = twoLookHint(cc);
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
		var method = options.method || 'cfop';
		var stage = getStage(cc, method);
		if (method == 'roux') {
			if (stage != 'cmll') {
				return {stage: stage};
			}
			var cmll = solveCMLL(facelet);
			if (!cmll) {
				return {stage: 'cmll', kind: 'cmll', name: 'no-alg'};
			}
			return {
				stage: 'cmll',
				kind: cmll.kind,
				name: cmll.name,
				key: cmll.key,
				base: cmll.base,
				pre: cmll.pre,
				post: cmll.post,
				moves: cmll.moves,
				alg: cmll.alg
			};
		}
		if (method == 'lbl' && stage != 'cross' && stage != 'solved') {
			var lbl = solveLbl(facelet, stage, options);
			if (!lbl) {
				return {stage: stage, kind: lblKind(stage), name: 'no-alg'};
			}
			return {
				stage: stage,
				kind: lbl.kind,
				name: lbl.name,
				key: lbl.key,
				base: lbl.base,
				pre: lbl.pre,
				post: lbl.post,
				moves: lbl.moves,
				alg: lbl.alg,
				pieces: lbl.pieces,
				pair: lbl.pair,
				protect: lbl.protect,
				distance: lbl.distance,
				corner: lbl.corner,
				edge: lbl.edge,
				cornerAt: lbl.cornerAt,
				edgeAt: lbl.edgeAt,
				cornerColors: lbl.cornerColors,
				edgeColors: lbl.edgeColors
			};
		}
		if (stage == 'cross') {
			var cross = solveCross(facelet, options);
			if (!cross || !cross.length) {
				return {stage: 'cross', name: 'no-alg'};
			}
			return {
				stage: 'cross',
				kind: 'cross',
				name: 'cross',
				key: '',
				base: cross.slice(),
				pre: '',
				post: '',
				moves: cross,
				alg: cross.join(' ')
			};
		}
		if (stage == 'f2l') {
			var pair = minPair(facelet);
			if (!pair) {
				return {stage: 'f2l', name: 'no-solution'};
			}
			return {
				stage: 'f2l',
				kind: 'f2l',
				pair: pair.pair,
				name: pair.name,
				distance: pair.distance,
				pieces: pair.pieces,
				key: '',
				base: [],
				pre: '',
				post: '',
				moves: [],
				alg: '',
				slot: pair.slot,
				corner: pair.corner,
				edge: pair.edge
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
		if (hint.kind == 'coll') {
			return isF2LDone(cc) && edgesOriented(cc) && cornersSolvedOnTop(cc);
		}
		if (hint.kind == 'cmll') {
			return isF2BDone(cc) && cornersSolvedOnTop(cc);
		}
		if (hint.kind == 'lblcorner') {
			if (!isCrossSolved(cc) || cc.ca[4 + hint.pair] != 4 + hint.pair) {
				return false;
			}
			for (var lc = 0; lc < 4; lc++) {
				if ((hint.protect & (1 << lc)) && cc.ca[4 + lc] != 4 + lc) {
					return false;
				}
			}
			return true;
		}
		if (hint.kind == 'lblmid') {
			if (!isCrossSolved(cc)) {
				return false;
			}
			for (var lm = 0; lm < 4; lm++) {
				if (cc.ca[4 + lm] != 4 + lm) {
					return false;
				}
			}
			if (cc.ea[8 + hint.pair] != (8 + hint.pair) * 2) {
				return false;
			}
			for (var le = 0; le < 4; le++) {
				if ((hint.protect & (1 << le)) && cc.ea[8 + le] != (8 + le) * 2) {
					return false;
				}
			}
			return true;
		}
		if (hint.kind == 'eoll') {
			return isF2LDone(cc) && edgesOriented(cc);
		}
		if (hint.kind == 'ocll') {
			return isF2LDone(cc) && edgesOriented(cc) && cornersOriented(cc);
		}
		if (hint.kind == 'cpll') {
			return isF2LDone(cc) && edgesOriented(cc) && cornersSolvedOnTop(cc);
		}
		if (hint.kind == 'epll') {
			return cc.isEqual(mathlib.CubieCube.SOLVED);
		}
		return cc.isEqual(mathlib.CubieCube.SOLVED);
	}

	/** cheap per-move snapshot used by the technique tips */
	function techniqueState(facelet) {
		var cc = cubieFromFacelet(facelet);
		return {
			eo: llEdgesOriented(cc),
			co: llCornersOriented(cc),
			cp: cornersPermuted(cc),
			solved: cc.isEqual(mathlib.CubieCube.SOLVED)
		};
	}

	var PAIR_KEYS = ['', 'xc', 'xxc', 'xxxc', 'xxxxc'];
	// a stage finished and the next move came within this pause, i.e. the
	// solver went straight on instead of stopping to look for the next stage
	var FLOW_MS = 520;

	/**
	 * State machine for the advanced-technique achievements.  It consumes the
	 * cf4op progress of every move, plus the last-layer snapshot while the
	 * solve is close to the end, and reports the techniques just completed.
	 * Keeping it pure makes every rule unit-testable without a cube UI.
	 *
	 * step() input: {prog, eo, co, cp, solved}
	 *   prog: 7 nothing, 6 cross, 5..2 F2L pairs, 1 OLL, 0 solved
	 *   eo/co/cp/solved are only read while prog <= 3
	 */
	function makeTechniqueTracker() {
		var st;

		function reset(seedProg) {
			st = {
				prev: (typeof seedProg == 'number') ? seedProg : -1,
				cross: false,
				pairs: 0,
				pair3: false,
				eo3: false,
				co3: false,
				f2l: false,
				llEo: false,
				llCo: false,
				llCp: false,
				oriented: false,
				orientedCp: false,
				pendingTip: '',
				llTipDone: false,
				ollFlowMarked: false,
				mark: null,
				bestProg: 7,
				done: false
			};
			if (st.prev >= 0) {
				st.bestProg = st.prev;
			}
			if (st.prev >= 2 && st.prev <= 6) {
				st.cross = true;
				st.pairs = 6 - st.prev;
			}
		}
		reset();

		function step(s) {
			var ev = [];
			var prog = (s && typeof s.prog == 'number') ? s.prog : -1;
			if (prog < 0 || st.done) {
				return ev;
			}
			var prev = st.prev;

			// ---- connected stages
			if (st.mark) {
				var mark = st.mark;
				st.mark = null;
				if (typeof s.t == 'number' && typeof mark.t == 'number' &&
						s.t >= mark.t && s.t - mark.t <= FLOW_MS &&
						(mark.key != 'flowop' || prog > 0)) {
					ev.push({key: mark.key});
				}
			}

			// A lagging COLL/OLLCP completion is only accepted when a later
			// state still shows an oriented last layer with the corners
			// placed, i.e. an EPLL state.  A normal PLL also passes through
			// such a state near its end, but it either ends there or breaks
			// the placement again, so the candidate is dropped.
			if (st.pendingTip) {
				var pending = st.pendingTip;
				if (prog > 3) {
					// the last layer cannot be read while an algorithm breaks
					// the F2L, so hold the candidate until the next state that
					// actually shows the last layer
				} else {
					st.pendingTip = '';
					if (prog > 0 && s && s.eo && s.co && s.cp) {
						st.oriented = true;
						st.orientedCp = true;
						st.llTipDone = true;
						ev.push({key: pending});
					} else if (prog == 0) {
						st.llTipDone = true;
					}
				}
			}

			if (!st.cross && prog <= 6) {
				st.cross = true;
				st.pairs = Math.min(4, 6 - prog);
				// only a real 7 -> 5/4/3/2 transition is an extended cross;
				// an unknown baseline must not invent one
				if (prev > 6 && st.pairs > 0) {
					ev.push({key: PAIR_KEYS[st.pairs]});
				}
			} else if (st.cross && prev > 6 && prog <= 6) {
				// the cross was broken by an F2L alg and is back: resync
				st.pairs = 6 - prog;
			} else if (!st.f2l && !st.pair3 && st.cross &&
					prev >= 2 && prev <= 6 && prog >= 2 && prog <= 6) {
				var nowPairs = 6 - prog;
				if (nowPairs - st.pairs >= 2) {
					ev.push({key: 'multi', n: nowPairs - st.pairs});
				}
				st.pairs = nowPairs;
			}

			// remember the last-layer state right after the third pair, i.e.
			// before the work on the last slot starts
			if (st.cross && prog == 3 && !st.pair3) {
				st.pair3 = true;
				st.eo3 = !!s.eo;
				st.co3 = !!s.co;
			}

			if (!st.f2l && st.cross && prev > 2 && prog <= 2) {
				st.f2l = true;
				st.llEo = !!s.eo;
				st.llCo = !!s.co;
				st.llCp = !!s.cp;
				if (st.pair3) {
					if (!st.eo3 && st.llEo && !st.llCo) {
						ev.push({key: 'zbls'});
					}
					if (!st.eo3 && st.llEo && st.llCo) {
						// the last pair finished the whole last-layer
						// orientation (edges and corners) while inserting
						ev.push({key: 'ols'});
					}
					if (st.eo3 && st.llCo && !st.co3) {
						ev.push({key: 'wv'});
					}
					if (st.eo3 && st.llCo && st.co3) {
						ev.push({key: 'ollskip'});
					}
				}
				st.oriented = st.llEo && st.llCo;
				st.orientedCp = st.llCp;
			} else if (st.f2l && prog > 0 && s.eo && s.co) {
				if (!st.oriented) {
					// first orientation-complete state after F2L.  When the
					// corners are already placed as well, the algorithm
					// solved the corner permutation too: COLL if the top
					// cross was done after the last F2L pair, OLLCP
					// otherwise.  Corners still unpermuted here just means a
					// normal OLL followed by PLL.
					st.oriented = true;
					st.orientedCp = !!s.cp;
					if (prog == 1 && st.orientedCp && !st.llCp &&
							!(st.llEo && st.llCo)) {
						st.llTipDone = true;
						ev.push({key: st.llEo ? 'coll' : 'ollcp'});
					}
				} else if (prog == 1 && s.cp && !st.orientedCp &&
						!st.llTipDone && !st.llCp &&
						!(st.llEo && st.llCo)) {
					// OLL algorithms finish the orientation a move or two
					// before the corner permutation.  Hold the tip until the
					// next state proves that an EPLL state follows; a normal
					// PLL's final G-perm step jumps straight to solved.
					st.pendingTip = st.llEo ? 'coll' : 'ollcp';
				}
			}

			if (prog == 0) {
				st.done = true;
				if (st.f2l) {
					if (prev > 2) {
						ev.push({key: 'llskip'});
					} else if (st.llEo && st.llCo) {
						// the last layer was already oriented after F2L: a
						// one-look PLL, not a ZBLL; when the corners were
						// placed as well, only the edges were left
						ev.push({key: st.llCp ? 'epll' : 'pll'});
					} else if (st.oriented && st.orientedCp) {
						// OLL (or COLL/OLLCP) finished with the corners
						// already placed, so the final algorithm only had to
						// permute the edges
						ev.push({key: 'epll'});
					} else if (!st.oriented) {
						ev.push({key: st.llEo ? 'zbll' : '1lll'});
					}
				}
			}

			st.prev = prog;

			// remember a finished stage: the next sampled state is the first
			// move of the following stage, so its timestamp tells the pause
			if (prog < st.bestProg) {
				var drop = st.bestProg - prog;
				st.bestProg = prog;
				if (drop == 1 && prog >= 1 && prog <= 6 &&
						typeof s.t == 'number') {
					st.mark = {
						t: s.t,
						key: prog == 6 ? 'flowcf' :
							(prog >= 3 ? 'flowff' : 'flowfl')
					};
				}
			}
			if (!st.ollFlowMarked && prog == 1 && s.eo && s.co &&
					typeof s.t == 'number') {
				st.ollFlowMarked = true;
				if (!st.mark) {
					st.mark = {t: s.t, key: 'flowop'};
				}
			}
			return ev;
		}

		return {
			reset: reset,
			step: step,
			state: function() {
				return st;
			}
		};
	}

	var ACHIEVEMENT_KEYS = ['xc', 'xxc', 'xxxc', 'xxxxc', 'zbls', 'wv', 'ols',
		'ollskip', 'zbll', 'coll', 'pll', 'epll', 'ollcp', 'llll', 'llskip',
		'multi', 'flowcf', 'flowff', 'flowfl', 'flowop', 'basicf2l'];
	// display order in the achievements panel
	var ACHIEVEMENT_ORDER = ['hot3', 'zb3', 'coll3', 'slot3', 'xc3',
		'flow3', 'basic3', 'fullhouse', 'triple', 'codex'];
	var ACHIEVEMENT_STREAKS = {
		hot3: 'hot', zb3: 'zbll', coll3: 'coll', slot3: 'slot', xc3: 'xc'
	};

	/**
	 * Cross-solve achievement state.  step() receives the set of technique
	 * keys earned in one solve and returns the achievements that were just
	 * unlocked.  The state is plain data so it can be stored in a csTimer
	 * property and round-tripped through JSON.
	 */
	function makeAchievementTracker() {
		var st;

		function freshState() {
			return {
				solves: 0,
				hot: 0,
				zbll: 0,
				coll: 0,
				slot: 0,
				xc: 0,
				seen: {},
				unlocked: {}
			};
		}

		function load(obj) {
			st = freshState();
			if (!obj || typeof obj != 'object') {
				return;
			}
			st.solves = Math.max(0, ~~obj.solves);
			st.hot = Math.max(0, ~~obj.hot);
			st.zbll = Math.max(0, ~~obj.zbll);
			st.coll = Math.max(0, ~~obj.coll);
			st.slot = Math.max(0, ~~obj.slot);
			st.xc = Math.max(0, ~~obj.xc);
			ACHIEVEMENT_KEYS.forEach(function(key) {
				if (obj.seen && obj.seen[key]) {
					st.seen[key] = Math.max(0, ~~obj.seen[key]);
				}
			});
			ACHIEVEMENT_ORDER.forEach(function(key) {
				if (obj.unlocked && obj.unlocked[key]) {
					st.unlocked[key] = 1;
				}
			});
		}

		function bump(name, ok) {
			st[name] = ok ? st[name] + 1 : 0;
		}

		function seenCount() {
			var n = 0;
			ACHIEVEMENT_KEYS.forEach(function(key) {
				if (st.seen[key]) {
					n++;
				}
			});
			return n;
		}

		function step(keys, stats) {
			var ev = [];
			var has = {};
			var uniq = [];
			(keys || []).forEach(function(key) {
				if (typeof key == 'string' && !has[key]) {
					has[key] = true;
					uniq.push(key);
					st.seen[key] = (st.seen[key] || 0) + 1;
				}
			});
			st.solves++;
			bump('hot', uniq.length > 0);
			bump('zbll', !!has.zbll);
			bump('coll', !!has.coll);
			bump('slot', !!(has.zbls || has.ols || has.wv));
			bump('xc', !!(has.xc || has.xxc || has.xxxc || has.xxxxc));

			function unlock(key, ok) {
				if (ok && !st.unlocked[key]) {
					st.unlocked[key] = 1;
					ev.push({key: key});
				}
			}
			// rarer / more specific achievements first so they lead the toast
			unlock('codex', seenCount() >= ACHIEVEMENT_KEYS.length);
			unlock('fullhouse', !!(has.zbls && has.zbll));
			unlock('triple', uniq.length >= 3);
			unlock('flow3', !!stats && ~~stats.flow >= 3);
			unlock('basic3', !!stats && ~~stats.basic >= 3);
			unlock('zb3', st.zbll >= 3);
			unlock('coll3', st.coll >= 3);
			unlock('slot3', st.slot >= 3);
			unlock('xc3', st.xc >= 3);
			unlock('hot3', st.hot >= 3);
			return ev;
		}

		function progress(key) {
			if (ACHIEVEMENT_STREAKS[key]) {
				var cur = st[ACHIEVEMENT_STREAKS[key]];
				return {cur: Math.min(cur, 3), target: 3};
			}
			if (key == 'codex') {
				return {cur: seenCount(), target: ACHIEVEMENT_KEYS.length};
			}
			return {cur: st.unlocked[key] ? 1 : 0, target: 1};
		}

		function serialize() {
			var seen = {};
			var unlocked = {};
			ACHIEVEMENT_KEYS.forEach(function(key) {
				if (st.seen[key]) {
					seen[key] = st.seen[key];
				}
			});
			ACHIEVEMENT_ORDER.forEach(function(key) {
				if (st.unlocked[key]) {
					unlocked[key] = 1;
				}
			});
			return {
				solves: st.solves,
				hot: st.hot,
				zbll: st.zbll,
				coll: st.coll,
				slot: st.slot,
				xc: st.xc,
				seen: seen,
				unlocked: unlocked
			};
		}

		load(null);
		return {
			load: load,
			step: step,
			state: function() {
				return st;
			},
			progress: progress,
			serialize: serialize,
			list: function() {
				return ACHIEVEMENT_ORDER.slice();
			},
			requiredKeys: function() {
				return ACHIEVEMENT_KEYS.slice();
			}
		};
	}

	return {
		getStage: function(facelet, method) {
			return getStage(cubieFromFacelet(facelet), method);
		},
		techniqueState: techniqueState,
		basicPair: basicPair,
		pairMask: pairMask,
		makeTechniqueTracker: makeTechniqueTracker,
		makeAchievementTracker: makeAchievementTracker,
		solveCross: solveCross,
		minPair: minPair,
		pairInfo: pairInfo,
		locatePair: locatePair,
		locateLbl: locateLbl,
		pairCount: pairCountOf,
		lblCornerHint: lblCornerHint,
		lblMidHint: lblMidHint,
		solveLbl: solveLbl,
		buildEollLut: buildEollLut,
		ensureEollLut: ensureEollLut,
		buildOcllLut: buildOcllLut,
		ensureOcllLut: ensureOcllLut,
		buildCpllLut: buildCpllLut,
		ensureCpllLut: ensureCpllLut,
		solveLastLayer: solveLastLayer,
		getHint: getHint,
		warmUp: warmUp,
		buildAlllLut: buildAlllLut,
		ensureAlllLut: ensureAlllLut,
		alllReady: alllReady,
		alllLutSize: alllLutSize,
		buildZbllLut: buildZbllLut,
		ensureZbllLut: ensureZbllLut,
		zbllReady: zbllReady,
		zbllLutSize: zbllLutSize,
		buildCollLut: buildCollLut,
		ensureCollLut: ensureCollLut,
		buildCmllLut: buildCmllLut,
		ensureCmllLut: ensureCmllLut,
		cmllLutSize: cmllLutSize,
		solveCMLL: solveCMLL,
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
