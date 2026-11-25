"use strict";

// Verify the Copilot engine:
//   1. The F2L hint highlights the corner and edge of the pair that needs the
//      fewest moves to insert without breaking the cross or a solved pair,
//      cross-checked by an independent exact search.
//   2. The ALLL LUT is complete (every entry solves its key) and we report how
//      much of the 62208 last-layer states it covers.
// Usage: node experiment/test_copilot.js [cases]

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var root = path.resolve(__dirname, '..');
var CASES = ~~process.argv[2] || 150;

function makeContext() {
	var ctx = {
		console: console,
		process: {env: {}},
		module: {exports: {}},
		Date: Date,
		Math: Math,
		JSON: JSON,
		setTimeout: setTimeout,
		self: {}
	};
	vm.createContext(ctx);
	vm.runInContext('var $={now:function(){return Date.now();},noop:function(){}};', ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/isaac.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/utillib.js'), 'utf8')
		.replace('const ISCSTIMER = true;', 'const ISCSTIMER = false;'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/mathlib.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/alll.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/ollpll.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/cmll.js'), 'utf8'), ctx);
	vm.runInContext(fs.readFileSync(path.join(root, 'src/js/lib/copilot.js'), 'utf8'), ctx);
	return ctx;
}

var ctx = makeContext();
var mathlib = ctx.mathlib;
var copilot = ctx.cubeCopilot;

var MOVE_NAMES = [];
for (var f = 0; f < 6; f++) {
	var face = 'URFDLB'.charAt(f);
	MOVE_NAMES.push(face, face + '2', face + "'");
}
var MOVE_CUBES = MOVE_NAMES.map(function(n) {
	var c = new mathlib.CubieCube();
	c.selfMoveStr(n);
	return c;
});

function mul(a, b) {
	var ret = new mathlib.CubieCube();
	mathlib.CubieCube.CubeMult(a, b, ret);
	return ret;
}

function applyMoves(cc, moves) {
	var cur = cc;
	for (var i = 0; i < moves.length; i++) {
		cur = mul(cur, MOVE_CUBES[moves[i]]);
	}
	return cur;
}

function applyAlg(cc, alg) {
	var cur = new mathlib.CubieCube();
	cur.init(cc.ca.slice(), cc.ea.slice());
	alg.split(/\s+/).filter(Boolean).forEach(function(t) {
		cur.selfMoveStr(t);
	});
	return cur;
}

function isPairSolved(cc, i) {
	var p = copilot.F2L_PAIRS[i];
	return (cc.ca[p.corner] & 7) == p.corner && (cc.ca[p.corner] >> 3) == 0 &&
		(cc.ea[p.edge] >> 1) == p.edge && (cc.ea[p.edge] & 1) == 0;
}

function crossSolved(cc) {
	for (var e = 4; e < 8; e++) {
		if ((cc.ea[e] >> 1) != e || (cc.ea[e] & 1) != 0) {
			return false;
		}
	}
	return true;
}

function stickerColors(facelet, idxs) {
	return idxs.map(function(i) {
		return facelet[i];
	}).sort().join('');
}

// ---- cross-solved position generator -------------------------------------
// BFS from the solved cube over the cross coordinate, storing a move path.
// Any random state can then be mapped to a cross-solved state by appending the
// path for its coordinate.

var CROSS_PATH = (function() {
	var CROSS_EDGES = [4, 5, 6, 7];
	// iterate over the cross pieces, not over slots: this coordinate is closed
	// under moves, so BFS distances are valid
	function coord(cc) {
		var out = [];
		for (var i = 0; i < CROSS_EDGES.length; i++) {
			var piece = CROSS_EDGES[i];
			for (var p = 0; p < 12; p++) {
				if ((cc.ea[p] >> 1) == piece) {
					out.push(p + ':' + (cc.ea[p] & 1));
					break;
				}
			}
		}
		return out.join('|');
	}
	var solved = new mathlib.CubieCube();
	var table = {};
	table[coord(solved)] = [];
	var frontier = [solved];
	var depth = 0;
	while (frontier.length && depth < 9) {
		var next = [];
		for (var i = 0; i < frontier.length; i++) {
			for (var m = 0; m < 18; m++) {
				var ns = mul(frontier[i], MOVE_CUBES[m]);
				var key = coord(ns);
				if (key in table) {
					continue;
				}
				table[key] = table[coord(frontier[i])].concat([m]);
				next.push(ns);
			}
		}
		frontier = next;
		depth++;
	}
	return {table: table, coord: coord};
})();

function crossSolvedFrom(cc) {
	var key = CROSS_PATH.coord(cc);
	var path = CROSS_PATH.table[key];
	if (!path) {
		return null;
	}
	// path goes solved -> key; to solve cc we need the reverse inverse
	var cur = cc;
	for (var i = path.length - 1; i >= 0; i--) {
		var m = path[i];
		var inv = Math.floor(m / 3) * 3 + (2 - m % 3);
		cur = mul(cur, MOVE_CUBES[inv]);
	}
	return cur;
}

function randomScramble(n) {
	var ret = [];
	var last = -1;
	for (var i = 0; i < n; i++) {
		var m;
		do {
			m = Math.floor(Math.random() * 18);
		} while (Math.floor(m / 3) == last);
		last = Math.floor(m / 3);
		ret.push(m);
	}
	return ret;
}

var failures = 0;
function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
}

// ---- minimal pair (no search) reference ----------------------------------
// The engine ranks the four F2L pairs by the pruning distance of their
// corner/edge coordinates.  Rebuild that distance here with an independent
// BFS so the ranking can be cross-checked.

function pairCoord(cc, pairIdx) {
	var pair = copilot.F2L_PAIRS[pairIdx];
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

function refPairDistances(pairIdx) {
	var table = new Int8Array(576);
	table.fill(-1);
	var solved = new mathlib.CubieCube();
	table[pairCoord(solved, pairIdx)] = 0;
	var frontier = [solved];
	var depth = 0;
	while (frontier.length && depth < 20) {
		depth++;
		var next = [];
		for (var f = 0; f < frontier.length; f++) {
			for (var m = 0; m < 18; m++) {
				var ns = mul(frontier[f], MOVE_CUBES[m]);
				var key = pairCoord(ns, pairIdx);
				if (table[key] >= 0) {
					continue;
				}
				table[key] = depth;
				next.push(ns);
			}
		}
		frontier = next;
	}
	return table;
}

var REF_PAIR_DIST = [refPairDistances(0), refPairDistances(1),
	refPairDistances(2), refPairDistances(3)];

// independent cross-distance table, used to prune the reference pair search
var REF_CROSS = (function() {
	function key(cc) {
		var ret = 0;
		for (var i = 0; i < 4; i++) {
			var piece = 4 + i;
			for (var p = 0; p < 12; p++) {
				if ((cc.ea[p] >> 1) == piece) {
					ret = ret * 24 + p * 2 + (cc.ea[p] & 1);
					break;
				}
			}
		}
		return ret;
	}
	var table = new Int8Array(331776);
	table.fill(-1);
	var solved = new mathlib.CubieCube();
	table[key(solved)] = 0;
	var frontier = [solved];
	var depth = 0;
	while (frontier.length && depth < 16) {
		depth++;
		var next = [];
		for (var f = 0; f < frontier.length; f++) {
			for (var m = 0; m < 18; m++) {
				var ns = mul(frontier[f], MOVE_CUBES[m]);
				var k = key(ns);
				if (table[k] >= 0) {
					continue;
				}
				table[k] = depth;
				next.push(ns);
			}
		}
		frontier = next;
	}
	return {key: key, table: table};
})();

function refLowerBound(cc, pairIdx, protectMask) {
	var h = REF_PAIR_DIST[pairIdx][pairCoord(cc, pairIdx)];
	var c = REF_CROSS.table[REF_CROSS.key(cc)];
	if (c < 0) {
		c = 99;
	}
	if (c > h) {
		h = c;
	}
	for (var p = 0; p < 4; p++) {
		if (!(protectMask & (1 << p)) || p == pairIdx) {
			continue;
		}
		var d = REF_PAIR_DIST[p][pairCoord(cc, p)];
		if (d > h) {
			h = d;
		}
	}
	return h;
}

function refGoal(cc, pairIdx, protectMask) {
	if (!isPairSolved(cc, pairIdx) || !crossSolved(cc)) {
		return false;
	}
	for (var p = 0; p < 4; p++) {
		if ((protectMask & (1 << p)) && !isPairSolved(cc, p)) {
			return false;
		}
	}
	return true;
}

function refSearchPair(cc, pairIdx, depth, protectMask) {
	var path = [];
	function dfs(state, remaining, lastFace) {
		if (refLowerBound(state, pairIdx, protectMask) > remaining) {
			return null;
		}
		if (remaining == 0) {
			return refGoal(state, pairIdx, protectMask) ? path.slice() : null;
		}
		for (var m = 0; m < 18; m++) {
			var face = ~~(m / 3);
			if (face == lastFace) {
				continue;
			}
			var ns = mul(state, MOVE_CUBES[m]);
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

// exact reference: shortest insertion while keeping the cross and every
// already solved pair intact
function refMinPair(cc, maxDepth) {
	var protect = 0;
	for (var p = 0; p < 4; p++) {
		if (isPairSolved(cc, p)) {
			protect |= 1 << p;
		}
	}
	if (!crossSolved(cc) || protect == 0xf) {
		return null;
	}
	for (var depth = 1; depth <= maxDepth; depth++) {
		for (var p2 = 0; p2 < 4; p2++) {
			if (protect & (1 << p2)) {
				continue;
			}
			if (refLowerBound(cc, p2, protect) > depth) {
				continue;
			}
			var path = refSearchPair(cc, p2, depth, protect);
			if (path) {
				return {pair: p2, distance: depth, moves: path};
			}
		}
	}
	return null;
}

var failures = 0;
function fail(msg) {
	failures++;
	console.log('FAIL ' + msg);
}

// ---- 1. F2L --------------------------------------------------------------

var f2lTests = 0;
var f2lMoves = 0;
var attempts = 0;

while (f2lTests < CASES && attempts < 20000) {
	attempts++;
	var cc = new mathlib.CubieCube();
	cc = applyMoves(cc, randomScramble(20));
	var pos = crossSolvedFrom(cc);
	if (!pos) {
		continue;
	}
	var facelet = pos.toFaceCube();
	if (copilot.getStage(facelet) != 'f2l') {
		continue;
	}
	f2lTests++;
	var hint = copilot.minPair(facelet);
	if (!hint) {
		fail('no F2L hint');
		continue;
	}
	if (hint.pair < 0 || hint.pair > 3) {
		fail('bad pair index ' + hint.pair);
		continue;
	}
	if (hint.pieces.length != 5) {
		fail('highlight should have 3 corner + 2 edge stickers, got ' + hint.pieces.length);
	}
	var seen = {};
	for (var i = 0; i < hint.pieces.length; i++) {
		var idx = hint.pieces[i];
		if (!(idx >= 0 && idx < 54) || seen[idx]) {
			fail('bad highlighted facelet index ' + idx);
		}
		seen[idx] = 1;
	}
		var unsolvedCount = 0;
		for (var up = 0; up < 4; up++) {
			if (!isPairSolved(pos, up)) unsolvedCount++;
		}
		if (unsolvedCount == 1) {
			if (hint.exact === true || hint.moves) {
				fail('last-pair hint should skip the exact search');
			}
			if (isPairSolved(pos, hint.pair)) {
				fail('last-pair hint points at a solved pair');
			}
		} else {
			if (hint.exact !== true || !hint.moves) {
				fail('F2L hint was not an exact search');
				continue;
			}
			if (hint.distance != hint.moves.length) {
				fail('F2L distance does not match its move sequence');
			}
			var after = applyAlg(pos, hint.moves.join(' '));
			if (!isPairSolved(after, hint.pair) || !crossSolved(after)) {
				fail('F2L hint does not insert the pair while keeping the cross');
			}
			for (var p = 0; p < 4; p++) {
				if (isPairSolved(pos, p) && !isPairSolved(after, p)) {
					fail('F2L hint broke an already solved pair');
				}
			}
			var ref = refMinPair(pos, 11);
			if (!ref || ref.pair != hint.pair || ref.distance != hint.distance) {
				fail('cheapest pair is ' + (ref && ref.pair) + ' in ' +
					(ref && ref.distance) + ' moves, got ' + hint.pair + ' in ' + hint.distance);
			}
		}
	// turning the whole cube must not swap the highlighted pair
	for (var r = 0; r < 24; r++) {
		var turned = copilot.orientFacelet(facelet, r);
		var same = copilot.locatePair(turned, hint.cornerColors, hint.edgeColors);
		if (!same || same.solved !== hint.solved) {
			fail('lost the hinted pair after rotation ' + r);
			continue;
		}
		if (stickerColors(turned, mathlib.CubieCube.cFacelet[same.cornerAt]) != hint.cornerColors ||
			stickerColors(turned, mathlib.CubieCube.eFacelet[same.edgeAt]) != hint.edgeColors) {
			fail('rotation ' + r + ' changed the highlighted pieces');
		}
	}
	f2lMoves += hint.distance;
}

console.log('F2L cases', f2lTests, 'avg min moves', (f2lMoves / Math.max(f2lTests, 1)).toFixed(2));

// the last remaining pair is highlighted without running the exact search
var lastPair = new mathlib.CubieCube();
['R', 'U', "R'"].forEach(function(t) {
	lastPair.selfMoveStr(t);
});
var lastPairFacelet = lastPair.toFaceCube();
var lastHint = copilot.minPair(lastPairFacelet);
if (copilot.getStage(lastPairFacelet) != 'f2l') {
	fail('last-pair setup is not an F2L case');
}
if (!lastHint || lastHint.pair != 0) {
	fail('last-pair hint picked the wrong pair');
}
if (lastHint && (lastHint.exact === true || lastHint.moves)) {
	fail('last-pair hint ran the exact search');
}
if (lastHint && lastHint.pieces.length != 5) {
	fail('last-pair hint has no highlight');
}

// a position without a solved cross has no F2L pair hint
var loose = new mathlib.CubieCube();
loose = applyMoves(loose, randomScramble(20));
if (loose.toFaceCube() != mathlib.SOLVED_FACELET && !crossSolved(loose)) {
	if (copilot.minPair(loose.toFaceCube())) {
		fail('F2L hint returned before the cross is solved');
	}
}

// ---- 2. ALLL -------------------------------------------------------------

var stats = copilot.buildAlllLut();
var LL_STATES = 62208;
console.log('ALLL base algs', stats.entries, 'LUT states', stats.unique,
	'(exact coverage incl. AUF is checked by test_coverage.js)');

var lut = copilot.ensureAlllLut();
var verified = 0;
Object.keys(lut).forEach(function(key) {
	var entry = lut[key];
	var parts = key.split(',').map(Number);
	var cc = new mathlib.CubieCube();
	for (var i = 0; i < 4; i++) {
		cc.ca[i] = parts[i * 2] | (parts[i * 2 + 1] << 3);
	}
	for (var i = 4; i < 8; i++) {
		cc.ca[i] = i;
	}
	for (var i = 0; i < 4; i++) {
		cc.ea[i] = (parts[8 + i * 2] << 1) | parts[8 + i * 2 + 1];
	}
	for (var i = 4; i < 12; i++) {
		cc.ea[i] = i * 2;
	}
	var after = applyAlg(cc, entry.alg);
	if (after.toFaceCube() != mathlib.SOLVED_FACELET) {
		fail('LUT entry broken: ' + key + ' ' + entry.alg);
	}
	verified++;
});
console.log('LUT entries verified', verified);

// ---- 3. Layer by layer ---------------------------------------------------

function edgesOriented(cc) {
	for (var i = 0; i < 4; i++) {
		if (cc.ea[i] & 1) {
			return false;
		}
	}
	return true;
}

function cornersOriented(cc) {
	for (var i = 0; i < 4; i++) {
		if (cc.ca[i] >> 3) {
			return false;
		}
	}
	return true;
}

function cornersPlaced(cc) {
	for (var i = 0; i < 4; i++) {
		if ((cc.ca[i] & 7) != i) {
			return false;
		}
	}
	return true;
}

// exhaustive: every legal last-layer state falls in the right beginner step
// and gets a verified algorithm for that step
(function() {
	function perms(n) {
		var out = [];
		var a = [];
		for (var i = 0; i < n; i++) {
			a.push(i);
		}
		(function rec(k) {
			if (k == n) {
				out.push(a.slice());
				return;
			}
			for (var i = k; i < n; i++) {
				var t = a[k];
				a[k] = a[i];
				a[i] = t;
				rec(k + 1);
				t = a[k];
				a[k] = a[i];
				a[i] = t;
			}
		})(0);
		return out;
	}
	function parity(a) {
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
	var P = perms(4);
	var states = 0;
	var staged = {eoll: 0, ocll: 0, cpll: 0, epll: 0, solved: 0};
	var miss = 0;
	P.forEach(function(cp) {
		for (var c0 = 0; c0 < 3; c0++) {
			for (var c1 = 0; c1 < 3; c1++) {
				for (var c2 = 0; c2 < 3; c2++) {
					for (var c3 = 0; c3 < 3; c3++) {
						var co = [c0, c1, c2, c3];
						if ((c0 + c1 + c2 + c3) % 3) {
							continue;
						}
						P.forEach(function(ep) {
							if (parity(cp) != parity(ep)) {
								return;
							}
							for (var e0 = 0; e0 < 2; e0++) {
								for (var e1 = 0; e1 < 2; e1++) {
									for (var e2 = 0; e2 < 2; e2++) {
										for (var e3 = 0; e3 < 2; e3++) {
											var eo = [e0, e1, e2, e3];
											if ((e0 + e1 + e2 + e3) % 2) {
												continue;
											}
											var ca = [];
											var ea = [];
											for (var i = 0; i < 8; i++) {
												ca.push(i);
											}
											for (i = 0; i < 12; i++) {
												ea.push(i * 2);
											}
											for (i = 0; i < 4; i++) {
												ca[i] = cp[i] | (co[i] << 3);
												ea[i] = (ep[i] << 1) | eo[i];
											}
											var cc = new mathlib.CubieCube();
											cc.init(ca, ea);
											var facelet = cc.toFaceCube();
											var expect = !edgesOriented(cc) ? 'eoll' :
												!cornersOriented(cc) ? 'ocll' :
													!cornersPlaced(cc) ? 'cpll' : 'epll';
											if (cc.toFaceCube() == mathlib.SOLVED_FACELET) {
												expect = 'solved';
											}
											var stage = copilot.getStage(facelet, 'lbl');
											if (stage != expect) {
												miss++;
												if (miss < 5) {
													fail('LBL stage ' + stage + ' expected ' + expect);
												}
												return;
											}
											staged[stage]++;
											states++;
											if (stage != 'solved') {
												var hint = copilot.getHint(facelet, {method: 'lbl', timeLimit: 100});
												if (!hint || !hint.moves || !hint.alg) {
													miss++;
													if (miss < 5) {
														fail('no LBL hint for a ' + stage + ' state');
													}
													return;
												}
												if (!copilot.validateAlg(facelet, hint.moves, hint)) {
													miss++;
													if (miss < 5) {
														fail('LBL hint does not solve the ' + stage + ' state: ' + hint.alg);
													}
												}
											}
										}
									}
								}
							}
						});
					}
				}
			}
		}
	});
	console.log('LBL last-layer states', states, JSON.stringify(staged), 'misses', miss);
})();

// the second-layer table covers every slot and every place the edge can be
(function() {
	var cases = 0;
	for (var slot = 0; slot < 4; slot++) {
		var positions = [0, 1, 2, 3, 8, 9, 10, 11];
		for (var pi = 0; pi < positions.length; pi++) {
			for (var ori = 0; ori < 2; ori++) {
				var pos = positions[pi];
				if (pos == 8 + slot && ori == 0) {
					continue;
				}
				var ca = [];
				var ea = [];
				for (var i = 0; i < 8; i++) {
					ca.push(i);
				}
				for (i = 0; i < 12; i++) {
					ea.push(i * 2);
				}
				var displaced = ea[pos] >> 1;
				ea[pos] = ((8 + slot) << 1) | ori;
				if (displaced != 8 + slot) {
					ea[8 + slot] = displaced << 1;
				}
				var cc = new mathlib.CubieCube();
				cc.init(ca, ea);
				var facelet = cc.toFaceCube();
				if (copilot.getStage(facelet, 'lbl') != 'sl') {
					continue;
				}
				var unsolved = [];
				for (i = 0; i < 4; i++) {
					if (cc.ea[8 + i] != (8 + i) * 2) {
						unsolved.push(i);
					}
				}
				var hint = copilot.lblMidHint(facelet, {timeLimit: 200});
				if (!hint || unsolved.indexOf(hint.pair) < 0) {
					fail('second-layer case ' + slot + '/' + pos + '/' + ori +
						' picked slot ' + (hint && hint.pair));
				} else if (!copilot.validateAlg(facelet, hint.moves, hint)) {
					fail('second-layer case ' + slot + '/' + pos + '/' + ori +
						' alg is invalid: ' + hint.alg);
				}
				cases++;
			}
		}
	}
	console.log('LBL second-layer cases', cases);
})();

// a full layer-by-layer solve: every hint validates and the stages progress
(function() {
	var solves = 0;
	var hints = 0;
	var moves = 0;
	var worst = 0;
	var longest = 0;
	for (var trial = 0; trial < 20; trial++) {
		var cc = applyMoves(new mathlib.CubieCube(), randomScramble(25));
		var cross = copilot.solveCross(cc.toFaceCube(), {timeLimit: 3000});
		if (!cross) {
			fail('LBL: could not solve the cross');
			continue;
		}
		cc = applyAlg(cc, cross.join(' '));
		var guard = 0;
		var done = false;
		while (guard++ < 40) {
			var facelet = cc.toFaceCube();
			var stage = copilot.getStage(facelet, 'lbl');
			if (stage == 'solved') {
				done = true;
				break;
			}
			if (stage == 'cross') {
				fail('LBL: cross broke during the solve');
				break;
			}
			var t0 = Date.now();
			var hint = copilot.getHint(facelet, {method: 'lbl', timeLimit: 4000});
			var dt = Date.now() - t0;
			if (dt > worst) {
				worst = dt;
			}
			if (!hint || !hint.moves || !hint.alg) {
				fail('LBL: no hint at stage ' + stage);
				break;
			}
			if (!copilot.validateAlg(facelet, hint.moves, hint)) {
				fail('LBL: hint does not apply at stage ' + stage + ': ' + hint.alg);
				break;
			}
			if (hint.moves.length > longest) {
				longest = hint.moves.length;
			}
			moves += hint.moves.length;
			hints++;
			cc = applyAlg(cc, hint.moves.join(' '));
		}
		if (done) {
			solves++;
		} else if (guard >= 40) {
			fail('LBL: solve did not finish');
		}
	}
	console.log('LBL solves', solves, 'hints', hints, 'avg hint moves',
		(hints ? (moves / hints).toFixed(1) : 0), 'longest', longest, 'slowest ms', worst);
})();

console.log(failures == 0 ? 'ALL OK' : ('FAILURES: ' + failures));
process.exitCode = failures == 0 ? 0 : 1;
