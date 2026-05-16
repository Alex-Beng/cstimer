"use strict";

var RemoteCube = (function() {
	var baseUrl = '';

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
		if (!baseUrl || moves.length == 0) return;
		$.ajax({
			url: baseUrl + '/api/moves',
			type: 'POST',
			contentType: 'application/json',
			data: JSON.stringify(moves),
			timeout: 5000,
			error: function(xhr, status, err) {
				DEBUG && console.log('[remotecube] sendMoves failed:', err || status);
			}
		});
	}

	return {
		connect: connect,
		sendMoves: sendMoves
	};
})();
