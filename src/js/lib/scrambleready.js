"use strict";

/**
 * Scramble readiness state machine.
 *
 * The scramble text arrives asynchronously (possibly generated in a worker),
 * while the user can press space at any time.  Reading the scramble before it
 * is refreshed would silently reuse the previous one, which is the bug this
 * module exists to prevent.
 *
 * Kept DOM-free so it can be unit tested; timer/virtual.js drives it.
 */
function makeScrambleGate() {
	var generating = false;
	var generation = 0;

	/** A new scramble generation started; the current text is stale. */
	function beginGenerating() {
		generating = true;
		generation++;
	}

	/** A scramble (or an empty placeholder) has been published. */
	function publish() {
		generating = false;
		generation++;
	}

	/**
	 * True when the text currently held can be applied: no generation is in
	 * flight.  Callers must publish() for every beginGenerating().
	 */
	function isReady() {
		return !generating;
	}

	/** Monotonic counter, bumped whenever the scramble text changes meaning. */
	function token() {
		return generation;
	}

	function reset() {
		generating = false;
		generation++;
	}

	return {
		beginGenerating: beginGenerating,
		publish: publish,
		isReady: isReady,
		token: token,
		reset: reset
	};
}

var scrambleGate = execBoth(makeScrambleGate, makeScrambleGate);
