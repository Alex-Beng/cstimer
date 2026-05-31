"use strict";

var RemoteCube = (function() {
	var baseUrl = '';
	var port = null;
	var writer = null;
	var useSerial = typeof localStorage != 'undefined' && localStorage['remoteCubeMode'] == 'serial';
	var sendQueue = [];
	var sending = false;

	function connect(url, callback) {
		baseUrl = url.replace(/\/$/, '');
		$.ajax({
			url: baseUrl + '/api/facelets',
			type: 'GET',
			dataType: 'text',
			timeout: 5000,
			success: function(data) {
				callback(null, data.trim());
			},
			error: function(xhr, status, err) {
				callback(err || status, null);
			}
		});
	}

	function sendMoves(moves) {
		if (moves.length == 0) return;
		sendQueue.push(moves);
		drainQueue();
	}

	function drainQueue() {
		if (sending || sendQueue.length == 0) return;
		sending = true;
		var moves = sendQueue.shift();
		if (useSerial && writer) {
			var cmd = JSON.stringify(moves) + '\n';
			var encoder = new TextEncoder();
			writer.write(encoder.encode(cmd)).then(function() {
				sending = false;
				drainQueue();
			}).catch(function(err) {
				DEBUG && console.log('[remotecube] serial send failed:', err);
				sending = false;
				drainQueue();
			});
		} else if (baseUrl) {
			$.ajax({
				url: baseUrl + '/api/moves',
				type: 'POST',
				contentType: 'application/json',
				data: JSON.stringify(moves),
				timeout: 5000,
				complete: function() {
					sending = false;
					drainQueue();
				},
				error: function(xhr, status, err) {
					DEBUG && console.log('[remotecube] sendMoves failed:', err || status);
				}
			});
		} else {
			sending = false;
		}
	}

	function setMode(mode) {
		if (mode == 'serial' && !port) {
			return openSerial();
		} else if (mode == 'http') {
			disconnectSerial();
			useSerial = false;
			localStorage['remoteCubeMode'] = 'http';
			return Promise.resolve('http');
		}
		return Promise.resolve(useSerial ? 'serial' : 'http');
	}

	function openSerial() {
		if (!('serial' in navigator)) {
			return Promise.reject('Web Serial API 不可用');
		}
		return navigator.serial.requestPort().then(function(p) {
			port = p;
			return port.open({ baudRate: 115200 });
		}).then(function() {
			writer = port.writable.getWriter();
			useSerial = true;
			localStorage['remoteCubeMode'] = 'serial';
			return 'serial';
		}).catch(function(err) {
			DEBUG && console.log('[remotecube] serial open failed:', err);
			throw err;
		});
	}

	function disconnectSerial() {
		if (writer) {
			writer.releaseLock();
			writer = null;
		}
		if (port) {
			port.close();
			port = null;
		}
	}

	function getMode() {
		return useSerial ? 'serial' : 'http';
	}

	return {
		connect: connect,
		sendMoves: sendMoves,
		setMode: setMode,
		getMode: getMode
	};
})();
