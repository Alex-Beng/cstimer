execMain(function() {
	var _gatt;
	var _service_data;
	var _chrct_read;
	var _chrct_write;

	var SERVICE_UUID_DATA = '00000010-0000-fff7-fff6-fff5fff4fff0';
	var CHRCT_UUID_READ = '0000fff6-0000-1000-8000-00805f9b34fb';
	var CHRCT_UUID_WRITE = '0000fff5-0000-1000-8000-00805f9b34fb';

	var GAN251_CIC_LIST = mathlib.valuedArray(256, function(i) { return (i << 8) | 0x01 });

	var GAN251_BASE_KEY = [
		0x58, 0x98, 0x61, 0xfc, 0x1f, 0xec, 0xd7, 0x60,
		0x9f, 0x85, 0xd3, 0x62, 0xbe, 0x37, 0x17, 0x2c
	];

	var GAN251_BASE_IV = [
		0x7f, 0x61, 0xd0, 0x52, 0x75, 0xc1, 0x39, 0x52,
		0x08, 0x2e, 0x54, 0x1d, 0x8a, 0x78, 0x63, 0x4d
	];

	var FACE_MASK_TO_FACE = {
		0x02: 'U',
		0x20: 'R',
		0x08: 'F',
		0x01: 'D',
		0x10: 'L',
		0x04: 'B'
	};

	var deviceName = null;
	var deviceMac = null;
	var decoder = null;
	var cornerPermutation = [0, 1, 2, 3, 4, 5, 6, 7];
	var cornerOrientation = [0, 0, 0, 0, 0, 0, 0, 0];
	var edgePermutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
	var edgeOrientation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	var batteryLevel = 0;

	function macToReversedSalt(mac) {
		var parts = mac.split(':');
		if (parts.length !== 6) {
			return null;
		}
		var salt = [];
		for (var i = 5; i >= 0; i--) {
			salt.push(parseInt(parts[i], 16));
		}
		return salt;
	}

	function deriveKeyIv(mac) {
		var salt = macToReversedSalt(mac);
		if (!salt) {
			return null;
		}
		var key = [];
		var iv = [];
		for (var i = 0; i < 16; i++) {
			if (i < 6) {
				key[i] = (GAN251_BASE_KEY[i] + salt[i]) % 0xff;
				iv[i] = (GAN251_BASE_IV[i] + salt[i]) % 0xff;
			} else {
				key[i] = GAN251_BASE_KEY[i];
				iv[i] = GAN251_BASE_IV[i];
			}
		}
		return { key: key, iv: iv };
	}

	function decryptPacket(data, key, iv) {
		var decrypted = [];
		for (var i = 0; i < data.length; i++) {
			decrypted[i] = data[i];
		}

		if (decrypted.length > 16) {
			var offset = decrypted.length - 16;
			var block = decoder.decrypt(decrypted.slice(offset));
			for (var i = 0; i < 16; i++) {
				decrypted[i + offset] = block[i] ^ (~~iv[i]);
			}
		}

		decoder.decrypt(decrypted);
		for (var i = 0; i < 16; i++) {
			decrypted[i] ^= (~~iv[i]);
		}

		return decrypted;
	}

	function trimTrailingZeros(bytes) {
		var end = bytes.length;
		while (end > 0 && bytes[end - 1] === 0) {
			end--;
		}
		return bytes.slice(0, end);
	}

	function crc16CcittFalse(bytes) {
		var crc = 0xffff;
		for (var i = 0; i < bytes.length; i++) {
			crc ^= bytes[i] << 8;
			for (var j = 0; j < 8; j++) {
				crc = (crc & 0x8000) !== 0 ? (crc << 1) ^ 0x1021 : crc << 1;
				crc &= 0xffff;
			}
		}
		return crc & 0xffff;
	}

	function validateCrc16(decrypted) {
		if (decrypted.length < 3) {
			return false;
		}
		var body = decrypted.slice(0, -2);
		var expected = decrypted[decrypted.length - 2] | (decrypted[decrypted.length - 1] << 8);
		var computed = crc16CcittFalse(body);
		if (computed === expected) { giikerutil.log('[gan251cube] CRC OK'); return true; }
		return false;
	}

	function readBits(bytes, offset, length) {
		var value = 0;
		for (var i = 0; i < length; i++) {
			var bitOffset = offset + i;
			var byteIdx = Math.floor(bitOffset / 8);
			var byte = bytes[byteIdx] || 0;
			var bit = (byte >> (7 - (bitOffset % 8))) & 1;
			value = (value << 1) | bit;
		}
		return value;
	}

	function inferMissingCorner(firstSeven) {
		var used = {};
		for (var i = 0; i < firstSeven.length; i++) {
			used[firstSeven[i]] = true;
		}
		for (var value = 0; value < 8; value++) {
			if (!used[value]) {
				return value;
			}
		}
		return 7;
	}

	function applyMove(face, direction) {
		if (direction === 'unknown') {
			return;
		}
		var cc = new mathlib.CubieCube();
		for (var i = 0; i < 8; i++) {
			cc.ca[i] = cornerPermutation[i] * 3 + cornerOrientation[i];
		}
		for (var i = 0; i < 12; i++) {
			cc.ea[i] = edgePermutation[i] << 1 | edgeOrientation[i];
		}
		// selfMoveStr handles the internal encoding correctly.
		// Suffix already encodes direction: 'U ' (cw), "U'" (ccw), 'U2' (double).
		var moveStr = face + (direction === 'counterclockwise' ? "'" : direction === 'double' ? '2' : ' ');
		cc.selfMoveStr(moveStr);
		for (var i = 0; i < 8; i++) {
			cornerPermutation[i] = (cc.ca[i] / 3) | 0;
			cornerOrientation[i] = cc.ca[i] % 3;
		}
		for (var i = 0; i < 12; i++) {
			edgePermutation[i] = cc.ea[i] >> 1;
			edgeOrientation[i] = cc.ea[i] & 1;
		}
	}

	function decodeMovePacket(decrypted) {
		if (decrypted.length < 11) {
			return null;
		}
		var rawMoveByte = decrypted[8];
		var faceMask = rawMoveByte & 0x3f;
		var turnBits = (rawMoveByte >> 6) & 0x03;
		var face = FACE_MASK_TO_FACE[faceMask];
		
		if (!face) {
			return null;
		}

		var direction = 'unknown';
		if (turnBits === 0) {
			direction = 'clockwise';
		} else if (turnBits === 1) {
			direction = 'counterclockwise';
		} else if (turnBits === 2) {
			direction = 'double';
		}

		var suffix = direction === 'counterclockwise' ? "'" : direction === 'double' ? '2' : '';
		var notation = face + suffix;

		return {
			face: face,
			direction: direction,
			notation: notation
		};
	}

	function decodeStatePacket(decrypted) {
		if (decrypted.length < 14) {
			return null;
		}

		var dataLength = decrypted[1];
		var dataEnd = Math.min(2 + dataLength, decrypted.length);
		var cubiePayload = decrypted.slice(4, dataEnd);

		var firstSevenCorners = [];
		for (var i = 0; i < 7; i++) {
			firstSevenCorners.push(readBits(cubiePayload, i * 3, 3));
		}
		var cp = firstSevenCorners.slice();
		cp.push(inferMissingCorner(firstSevenCorners));

		// co: 8 values × 2 bits starting at offset 21
		var co = [];
		for (var i = 0; i < 8; i++) {
			co.push(readBits(cubiePayload, 21 + i * 2, 2));
		}

		// ep: 11 values × 4 bits starting at offset 21+16=37
		var firstElevenEdges = [];
		for (var i = 0; i < 11; i++) {
			firstElevenEdges.push(readBits(cubiePayload, 37 + i * 4, 4));
		}
		var ep = firstElevenEdges.slice();
		ep.push(inferMissingEdge(firstElevenEdges));

		// eo: 12 values × 1 bit starting at offset 37+44=81
		var eo = [];
		for (var i = 0; i < 12; i++) {
			eo.push(readBits(cubiePayload, 81 + i, 1));
		}

		return {
			cornerPermutation: cp,
			cornerOrientation: co,
			edgePermutation: ep,
			edgeOrientation: eo
		};
	}

	function inferMissingEdge(firstEleven) {
		var used = {};
		for (var i = 0; i < firstEleven.length; i++) {
			used[firstEleven[i]] = true;
		}
		for (var value = 0; value < 12; value++) {
			if (!used[value]) {
				return value;
			}
		}
		return 11;
	}

	function buildFacelet() {
		var C_FACELET = [[8,9,20],[6,18,38],[0,36,47],[2,45,11],[29,26,15],[27,44,24],[33,53,42],[35,17,51]];
		var E_FACELET = [[5,10],[7,19],[3,37],[1,46],[32,16],[28,25],[30,43],[34,52],[23,12],[21,41],[50,39],[48,14]];
		var C_COLOR = [[0,1,2],[0,2,4],[0,4,5],[0,5,1],[3,2,1],[3,4,2],[3,5,4],[3,1,5]];
		var E_COLOR = [[0,1],[0,2],[0,4],[0,5],[3,1],[3,2],[3,4],[3,5],[2,1],[2,4],[5,4],[5,1]];
		var cols = 'URFDLB';
		// Start with solved facelet (centers + edges correct), overwrite corners only
		var f = mathlib.SOLVED_FACELET.split('');
		// Match emulator: f[C_FACELET[i][(k+ori)%3]] = cols[C_COLOR[j][k]]
		for (var i = 0; i < 8; i++) {
			var j = cornerPermutation[i];
			var o = cornerOrientation[i];
			for (var k = 0; k < 3; k++) {
				f[C_FACELET[i][(k + o) % 3]] = cols.charAt(C_COLOR[j][k]);
			}
		}
		// Edges from tracked state
		for (var i = 0; i < 12; i++) {
			var j = edgePermutation[i];
			var o = edgeOrientation[i];
			for (var k = 0; k < 2; k++) {
				f[E_FACELET[i][(k + o) % 2]] = cols.charAt(E_COLOR[j][k]);
			}
		}
		var result = f.join('');
		// Verify
		var cnts = {};
		for (var c = 0; c < 54; c++) { var ch = result.charAt(c); cnts[ch] = (cnts[ch] || 0) + 1; }
		if (cnts.U != 9 || cnts.R != 9 || cnts.F != 9 || cnts.D != 9 || cnts.L != 9 || cnts.B != 9) {
			giikerutil.log('[gan251cube] BAD facelet:', JSON.stringify(cnts));
		}
		return result;
	}

	function processDecryptedPacket(decrypted) {
		if (decrypted.length < 1) {
			return;
		}

		var packetId = decrypted[0];
		var crcValid = validateCrc16(decrypted);

		if (packetId === 0x01) {
			if (!crcValid) {
				giikerutil.log('[gan251cube] CRC validation failed');
				return;
			}
			var moveData = decodeMovePacket(decrypted);
			if (moveData) {
				giikerutil.log('[gan251cube] Move:', moveData.notation);
				applyMove(moveData.face, moveData.direction);
				GiikerCube.callback(buildFacelet(), moveData.notation ? [moveData.notation] : [], [0, $.now()], deviceName);
			}
		} else if (packetId === 0xed) {
			if (!crcValid) {
				giikerutil.log('[gan251cube] CRC validation failed (state, processing anyway)');
			}
			var stateData = decodeStatePacket(decrypted);
			if (stateData) {
				giikerutil.log('[gan251cube] State update');
				cornerPermutation = stateData.cornerPermutation;
				cornerOrientation = stateData.cornerOrientation;
				edgePermutation = stateData.edgePermutation;
				edgeOrientation = stateData.edgeOrientation;
				giikerutil.log('[gan251cube] cp:', cornerPermutation.join(','), 'co:', cornerOrientation.join(','), 'ep:', edgePermutation.join(','), 'eo:', edgeOrientation.join(','));
				var fl = buildFacelet();
				giikerutil.log('[gan251cube] facelet:', fl);
				GiikerCube.callback(fl, [], [0, $.now()], deviceName);
			}
		} else if (packetId === 0xef) {
			if (!crcValid) {
				giikerutil.log('[gan251cube] CRC validation failed');
				return;
			}
			if (decrypted.length >= 3) {
				batteryLevel = decrypted[1];
				giikerutil.log('[gan251cube] Battery:', batteryLevel + '%');
				giikerutil.updateBattery([batteryLevel, deviceName + '*']);
			}
		} else if (!crcValid) {
			giikerutil.log('[gan251cube] CRC validation failed');
		}
	}

	function onStateChanged(event) {
		var value = event.target.value;
		var data = [];
		for (var i = 0; i < value.byteLength; i++) {
			data.push(value.getUint8(i));
		}

		if (!decoder || !deviceMac) {
			giikerutil.log('[gan251cube] No decoder available');
			return;
		}

		giikerutil.log('[gan251cube] raw:', data.map(function(b) { return ('0' + b.toString(16)).slice(-2); }).join(' '));
		var decrypted = decryptPacket(data, decoder.key, decoder.iv);
		giikerutil.log('[gan251cube] dec:', decrypted.map(function(b) { return ('0' + b.toString(16)).slice(-2); }).join(' '));
		decrypted = trimTrailingZeros(decrypted);
		processDecryptedPacket(decrypted);
	}

	function getMacFromAdv(mfData) {
		if (mfData instanceof DataView) {
			return new DataView(mfData.buffer.slice(2, 11));
		}
		for (var i = 0; i < GAN251_CIC_LIST.length; i++) {
			var id = GAN251_CIC_LIST[i];
			if (mfData.has(id)) {
				return new DataView(mfData.get(id).buffer.slice(0, 9));
			}
		}
		return null;
	}

	function parseMacBytes(macBytes) {
		var macParts = [];
		var len = macBytes.byteLength;
		for (var i = 0; i < 6; i++) {
			macParts.push(('0' + macBytes.getUint8(len - i - 1).toString(16)).slice(-2));
		}
		return macParts.join(':').toUpperCase();
	}

	function setupDecoder(mac) {
		deviceMac = mac.toUpperCase();
		giikerutil.log('[gan251cube] MAC:', deviceMac);
		var keyIv = deriveKeyIv(deviceMac);
		if (!keyIv) return false;
		giikerutil.log('[gan251cube] KEY:', keyIv.key.join(','));
		giikerutil.log('[gan251cube] IV :', keyIv.iv.join(','));
		decoder = $.aes128(keyIv.key);
		decoder.iv = keyIv.iv;
		return true;
	}

	function connectService() {
		return _gatt.getPrimaryService(SERVICE_UUID_DATA);
	}

	function showMacDialog() {
		return new Promise(function(resolve) {
			kernel.showDialog([$('<div>').append(
				$('<p>').text('无法获取 MAC。请在 chrome://bluetooth-internals/#devices 中找到设备 MAC 并输入:'),
				$('<input type="text" id="gan251MacInput" style="width:100%">').val('AB:12:34:5F:A4:CA')
			), function() {
				var mac = $('#gan251MacInput').val().trim().replace(/[^A-Fa-f0-9:]/g, '');
				resolve(mac || null);
			}, function() {
				resolve(null);
			}, 0], 'share', 'GAN251 MAC 输入');
		});
	}

	function init(device) {
		giikerutil.log('[gan251cube] init start');
		deviceName = device.name;

		return device.gatt.connect().then(function(gatt) {
			giikerutil.log('[gan251cube] gatt connected');
			_gatt = gatt;
			return GiikerCube.waitForAdvs().then(function(mfData) {
				var macBytes = getMacFromAdv(mfData);
				if (macBytes) {
					setupDecoder(parseMacBytes(macBytes));
					return Promise.resolve();
				}
				return Promise.reject('no mac');
			}).catch(function() {
				return showMacDialog().then(function(mac) {
					if (!mac) return Promise.reject(-1);
					setupDecoder(mac);
				});
			});
		}).then(function() {
			return connectService();
		}).then(function(service) {
			giikerutil.log('[gan251cube] got service');
			_service_data = service;
			return service.getCharacteristics();
		}).then(function(chrcts) {
			giikerutil.log('[gan251cube] got characteristics');
			_chrct_read = GiikerCube.findUUID(chrcts, CHRCT_UUID_READ);
			_chrct_write = GiikerCube.findUUID(chrcts, CHRCT_UUID_WRITE);
			if (!_chrct_read) {
				return Promise.reject('Cannot find read characteristic');
			}
			return _chrct_read.startNotifications();
		}).then(function() {
			giikerutil.log('[gan251cube] notifications started');
			_chrct_read.addEventListener('characteristicvaluechanged', onStateChanged);
			cornerPermutation = [0, 1, 2, 3, 4, 5, 6, 7];
			cornerOrientation = [0, 0, 0, 0, 0, 0, 0, 0];
			edgePermutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
			edgeOrientation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			return Promise.resolve();
		});
	}

	function getBatteryLevel() {
		return Promise.resolve(batteryLevel);
	}

	function clear() {
		var result = Promise.resolve();
		if (_chrct_read) {
			_chrct_read.removeEventListener('characteristicvaluechanged', onStateChanged);
			result = _chrct_read.stopNotifications().catch($.noop);
			_chrct_read = null;
		}
		_service_data = null;
		_chrct_write = null;
		_gatt = null;
		deviceName = null;
		deviceMac = null;
		decoder = null;
		cornerPermutation = [0, 1, 2, 3, 4, 5, 6, 7];
		cornerOrientation = [0, 0, 0, 0, 0, 0, 0, 0];
		edgePermutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
		edgeOrientation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		batteryLevel = 0;
		return result;
	}

	GiikerCube.regCubeModel({
		prefix: ['GAN251', 'gan251ui_', 'ganic251_', 'gan251ui'],
		init: init,
		opservs: [SERVICE_UUID_DATA],
		cics: GAN251_CIC_LIST,
		getBatteryLevel: getBatteryLevel,
		clear: clear,
		puzzleSize: 2
	});
});
