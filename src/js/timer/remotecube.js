"use strict";

execMain(function(timer) {
	var enable = false;
	var isConnected = false;
	var isConnecting = false;
	var puzzleObj;
	var div = $('<div />');
	var ori = 0;
	var moveQueue = [];
	var sendTimer = 0;
	var insTime = 0;
	var moveCnt = 0;
	var totPhases = 1;
	var rawMoves = [];
	var CubeMoveRE = /^\s*([URFDLB]w?|[EMSyxz]|2-2[URFDLB]w)(['2]?)\s*$/;

	function resetOri() {
		ori = 0;
	}

	function flushMoves() {
		sendTimer = 0;
		if (moveQueue.length == 0) return;
		RemoteCube.sendMoves(moveQueue);
		moveQueue = [];
	}

	function enqueueMove(face, dir) {
		moveQueue.push({face: face, dir: dir});
		clearTimeout(sendTimer);
		sendTimer = setTimeout(flushMoves, 100);
	}

	function decodeMoveIdx(idx) {
		var axis = Math.floor(idx / 3);
		var pow = idx % 3;
		if (pow == 1) {
			enqueueMove(axis, 0);
			enqueueMove(axis, 0);
		} else {
			enqueueMove(axis, pow == 2 ? 1 : 0);
		}
	}

	function processMoveForAPI(move) {
		var str = puzzleObj.move2str(move);
		var m = CubeMoveRE.exec(str);
		if (!m) return;
		var face = m[1];
		var pow = "2'".indexOf(m[2] || '-') + 2;

		var axis = 'URFDLB'.indexOf(face);
		if (axis != -1) {
			var moveIdx = axis * 3 + pow % 4 - 1;
			decodeMoveIdx(mathlib.CubieCube.rotMulM[ori][moveIdx]);
			return;
		}
		axis = 'UwRwFwDwLwBw'.indexOf(face);
		if (axis != -1) {
			axis >>= 1;
			var moveIdx = (axis + 3) % 6 * 3 + pow % 4 - 1;
			decodeMoveIdx(mathlib.CubieCube.rotMulM[ori][moveIdx]);
			var rot = [3, 15, 17, 1, 11, 23][axis];
			for (var i = 0; i < pow; i++) {
				ori = mathlib.CubieCube.rotMult[rot][ori];
			}
			return;
		}
		axis = 'yxz'.indexOf(face);
		if (axis != -1) {
			var rot = [3, 15, 17][axis];
			for (var i = 0; i < pow; i++) {
				ori = mathlib.CubieCube.rotMult[rot][ori];
			}
			return;
		}
		axis = ['2-2Uw', '2-2Rw', '2-2Fw', '2-2Dw', '2-2Lw', '2-2Bw'].indexOf(face);
		if (axis == -1) {
			axis = [null, null, 'S', 'E', 'M', null].indexOf(face);
		}
		if (axis != -1) {
			var m1 = axis * 3 + (4 - pow) % 4 - 1;
			var m2 = (axis + 3) % 6 * 3 + pow % 4 - 1;
			decodeMoveIdx(mathlib.CubieCube.rotMulM[ori][m1]);
			decodeMoveIdx(mathlib.CubieCube.rotMulM[ori][m2]);
			var rot = [3, 15, 17, 1, 11, 23][axis];
			for (var i = 0; i < pow; i++) {
				ori = mathlib.CubieCube.rotMult[rot][ori];
			}
			return;
		}
	}

	function moveListener(move, mstep, ts) {
		if (!isConnected) return;
		if (mstep == 1) return;
		var now = ts || $.now();
		if (timer.status() == -3 || timer.status() == -2) {
			if (puzzleObj.isRotation(move)) {
				if (mstep == 0) {
					rawMoves[0].push([puzzleObj.move2str(move), 0]);
				}
				return;
			} else {
				if (timer.checkUseIns()) {
					insTime = now - timer.startTime();
				} else {
					insTime = 0;
				}
				timer.startTime(now);
				moveCnt = 0;
				timer.curTime([insTime > 17000 ? -1 : (insTime > 15000 ? 2000 : 0)]);
				timer.status(cubeutil.getStepCount(kernel.getProp('vrcMP', 'n')));
				var inspectionMoves = rawMoves[0];
				rawMoves = [];
				for (var i = 0; i < timer.status(); i++) {
					rawMoves[i] = [];
				}
				rawMoves[timer.status()] = inspectionMoves;
				totPhases = timer.status();
				timer.updateMulPhase(totPhases, puzzleObj.isSolved(kernel.getProp('vrcMP', 'n')), now);
				timer.lcd.fixDisplay(false, true);
			}
		}
		if (timer.status() >= 1) {
			if (mstep == 0) {
				rawMoves[timer.status() - 1].push([puzzleObj.move2str(move), now - timer.startTime()]);
				processMoveForAPI(move);
			}
			var curProgress;
			if (mstep == 2) {
				curProgress = puzzleObj.isSolved(kernel.getProp('vrcMP', 'n'));
				timer.updateMulPhase(totPhases, curProgress, now);
			}
			if (mstep == 2 && curProgress == 0) {
				moveCnt += puzzleObj.moveCnt();
				flushMoves();
				timer.lcd.setStaticAppend('');
				timer.status(-1);
				$('#lcd').css({'visibility': 'unset'});
				timer.lcd.fixDisplay(false, true);
				rawMoves.reverse();
				kernel.pushSignal('time', ["", 0, timer.curTime(), 0, [$.map(rawMoves, cubeutil.moveSeq2str).filter($.trim).join(' '), '333', moveCnt]]);
			}
		}
	}

	function initPuzzle(callback) {
		if (isConnecting) return;
		isConnecting = true;
		puzzleObj = undefined;
		var options = {
			puzzle: 'cube3',
			style: 'v'
		};
		puzzleFactory.init(options, moveListener, div, function(ret, isInit) {
			puzzleObj = ret;
			isConnecting = false;
			if (!puzzleObj) {
				if (callback) callback(false);
				return;
			}
			puzzleObj.moveCnt(true);
			rawMoves = [[]];
			div.show();
			isConnected = true;
			timer.lcd.fixDisplay(false, true);
			setSize(kernel.getProp('timerSize'));
			if (callback) callback(true);
		});
	}

	function doScramble(facelets) {
		if (!puzzleObj) return;
		if (facelets != mathlib.SOLVED_FACELET) {
			var gen = scramble_333.genFacelet(facelets);
			if (gen) {
				var moves = puzzleObj.parseScramble(gen, true);
				puzzleObj.applyMoves(moves);
			}
		}
		puzzleObj.moveCnt(true);
		rawMoves = [[]];
	}

	function scrambleIt() {
		resetOri();
		moveQueue = [];
		if (sendTimer) { clearTimeout(sendTimer); sendTimer = 0; }
		RemoteCube.connect(remoteCubeUrl, function(err, facelets) {
			if (err) {
				isConnected = false;
				div.hide();
				timer.status(-1);
				timer.lcd.fixDisplay(false, true);
				return;
			}
			var now = $.now();
			if (!puzzleObj) {
				initPuzzle(function(ok) {
					if (!ok) return;
					doScramble(facelets);
					if (timer.checkUseIns()) {
						timer.startTime(now);
						timer.status(-3);
					} else {
						timer.lcd.val(0);
						timer.status(-2);
					}
					$('#lcd').css({'visibility': 'hidden'});
					timer.lcd.fixDisplay(false, true);
				});
				return;
			}
			isConnected = true;
			div.show();
			doScramble(facelets);
			if (timer.checkUseIns()) {
				timer.startTime(now);
				timer.status(-3);
			} else {
				timer.lcd.val(0);
				timer.status(-2);
			}
			$('#lcd').css({'visibility': 'hidden'});
			timer.lcd.fixDisplay(false, true);
		});
	}

	function showConnectDialog() {
		kernel.showDialog([$('<div>').append(
			$('<p>').text('输入远程魔方 URL:'),
			$('<input type="text" id="remoteCubeUrl" style="width:100%">').val(localStorage['remoteCubeUrl'] || 'http://')
		), function() {
			var url = $('#remoteCubeUrl').val();
			if (!url) return;
			localStorage['remoteCubeUrl'] = url;
			remoteCubeUrl = url;
			initPuzzle();
		}, 0, 0], 'share', '远程魔方连接');
	}

	function onkeydown(keyCode) {
		if (!isConnected || !puzzleObj) return;
		var now = $.now();
		if (timer.status() == -1) {
			if (keyCode == 32) {
				scrambleIt();
			}
		} else if (timer.status() == -3 || timer.status() == -2 || timer.status() >= 1) {
			if (keyCode == 27 || keyCode == 28) {
				flushMoves();
				var recordDNF = timer.status() >= 1;
				timer.lcd.setStaticAppend('');
				timer.status(-1);
				$('#lcd').css({'visibility': 'unset'});
				timer.lcd.fixDisplay(false, true);
				if (recordDNF) {
					rawMoves.reverse();
					kernel.pushSignal('time', ["", 0, [-1, now - timer.startTime()], 0, [$.map(rawMoves, cubeutil.moveSeq2str).filter($.trim).join(' '), '333', moveCnt]]);
				}
			} else {
				var mappedCode = help.getMappedCode(keyCode);
				puzzleObj.keydown({keyCode: mappedCode});
			}
		}
		if (keyCode == 27 || keyCode == 32) {
			kernel.clrKey();
		}
	}

	function onkeyup(keyCode) {}

	function setEnable(input) {
		var next = input == 'r';
		if (next == enable) return;
		enable = next;
		if (enable) {
			if (remoteCubeUrl) {
				initPuzzle();
			} else {
				showConnectDialog();
			}
		} else {
			isConnected = false;
			isConnecting = false;
			div.hide();
			puzzleObj = undefined;
			moveQueue = [];
			if (sendTimer) { clearTimeout(sendTimer); sendTimer = 0; }
		}
	}

	function setSize(value) {
		div.css('height', value * $('#logo').width() / 9 + 'px');
		puzzleObj && puzzleObj.resize();
	}

	var remoteCubeUrl = localStorage['remoteCubeUrl'] || '';

	$(function() {
		div.appendTo('#container');
		div.hide();
	});

	timer.remotecube = {
		setEnable: setEnable,
		onkeydown: onkeydown,
		onkeyup: onkeyup,
		setSize: setSize
	};
}, [timer]);
