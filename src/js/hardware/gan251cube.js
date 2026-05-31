execMain(function() {
	var _gatt;
	var _service_data;
	var _chrct_read;
	var _chrct_write;

	var SERVICE_UUID_DATA = '0000fff0-0000-1000-8000-00805f9b34fb';
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

	var MOVE_DEFS = {
		'U': { cycle: [0, 1, 2, 3], coDelta: [0, 0, 0, 0] },
		'R': { cycle: [0, 3, 7, 4], coDelta: [2, 1, 2, 1] },
		'F': { cycle: [0, 4, 5, 1], coDelta: [1, 2, 1, 2] },
		'D': { cycle: [4, 7, 6, 5], coDelta: [0, 0, 0, 0] },
		'L': { cycle: [1, 5, 6, 2], coDelta: [1, 2, 1, 2] },
		'B': { cycle: [2, 6, 7, 3], coDelta: [2, 1, 2, 1] }
	};

	var deviceName = null;
	var deviceMac = null;
	var decoder = null;
	var cornerPermutation = [0, 1, 2, 3, 4, 5, 6, 7];
	var cornerOrientation = [0, 0, 0, 0, 0, 0, 0, 0];
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
				decrypted[i + offset] = block[i] ^ (iv[i] || 0);
			}
		}
		
		var firstBlock = decoder.decrypt(decrypted.slice(0, 16));
		for (var i = 0; i < 16; i++) {
			decrypted[i] = firstBlock[i] ^ (iv[i] || 0);
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
		return computed === expected;
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

	function applyClockwiseFace(face) {
		var def = MOVE_DEFS[face];
		var oldCp = cornerPermutation.slice();
		var oldCo = cornerOrientation.slice();
		var cycle = def.cycle;
		var coDelta = def.coDelta;

		cornerPermutation[cycle[0]] = oldCp[cycle[3]];
		cornerPermutation[cycle[1]] = oldCp[cycle[0]];
		cornerPermutation[cycle[2]] = oldCp[cycle[1]];
		cornerPermutation[cycle[3]] = oldCp[cycle[2]];

		cornerOrientation[cycle[0]] = (oldCo[cycle[3]] + coDelta[0]) % 3;
		cornerOrientation[cycle[1]] = (oldCo[cycle[0]] + coDelta[1]) % 3;
		cornerOrientation[cycle[2]] = (oldCo[cycle[1]] + coDelta[2]) % 3;
		cornerOrientation[cycle[3]] = (oldCo[cycle[2]] + coDelta[3]) % 3;
	}

	function applyMove(face, direction) {
		var turns = direction === 'clockwise' ? 1 : direction === 'double' ? 2 : direction === 'counterclockwise' ? 3 : 0;
		for (var i = 0; i < turns; i++) {
			applyClockwiseFace(face);
		}
	}

	function decodeMovePacket(decrypted) {
		if (decrypted.length < 8) {
			return null;
		}
		var rawMoveByte = decrypted[5];
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
		if (decrypted.length < 18) {
			return null;
		}

		var dataLength = decrypted[1];
		var dataEnd = Math.min(2 + dataLength, decrypted.length - 2);
		var cubiePayload = decrypted.slice(4, dataEnd);

		var firstSevenCorners = [];
		for (var i = 0; i < 7; i++) {
			firstSevenCorners.push(readBits(cubiePayload, i * 3, 3));
		}
		
		var cp = firstSevenCorners.slice();
		cp.push(inferMissingCorner(firstSevenCorners));

		var co = [];
		for (var i = 0; i < 8; i++) {
			co.push(readBits(cubiePayload, 21 + i * 2, 2) % 3);
		}

		return {
			cornerPermutation: cp,
			cornerOrientation: co
		};
	}

	function processDecryptedPacket(decrypted) {
		if (decrypted.length < 1) {
			return;
		}

		var packetId = decrypted[0];
		var crcValid = validateCrc16(decrypted);

		if (!crcValid) {
			giikerutil.log('[gan251cube] CRC validation failed');
			return;
		}

		if (packetId === 0x01) {
			var moveData = decodeMovePacket(decrypted);
			if (moveData) {
				giikerutil.log('[gan251cube] Move:', moveData.notation);
				applyMove(moveData.face, moveData.direction);
				GiikerCube.callback(moveData.notation, [0, 0, 0, 0, 0, 0], deviceName);
			}
		} else if (packetId === 0xed) {
			var stateData = decodeStatePacket(decrypted);
			if (stateData) {
				giikerutil.log('[gan251cube] State update');
				cornerPermutation = stateData.cornerPermutation;
				cornerOrientation = stateData.cornerOrientation;
			}
		} else if (packetId === 0xef) {
			if (decrypted.length >= 3) {
				batteryLevel = decrypted[1];
				giikerutil.log('[gan251cube] Battery:', batteryLevel + '%');
				giikerutil.updateBattery([batteryLevel, deviceName + '*']);
			}
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

		var decrypted = decryptPacket(data, decoder.key, decoder.iv);
		decrypted = trimTrailingZeros(decrypted);
		processDecryptedPacket(decrypted);
	}

	function init(device) {
		giikerutil.log('[gan251cube] init start');
		deviceName = device.name;
		
		return device.gatt.connect().then(function(gatt) {
			giikerutil.log('[gan251cube] gatt connected');
			_gatt = gatt;
			return GiikerCube.waitForAdvs();
		}).then(function(mfData) {
			giikerutil.log('[gan251cube] got manufacturer data');
			var macBytes = null;
			
			if (mfData instanceof DataView) {
				macBytes = new DataView(mfData.buffer.slice(2, 11));
			} else {
				for (var i = 0; i < GAN251_CIC_LIST.length; i++) {
					var id = GAN251_CIC_LIST[i];
					if (mfData.has(id)) {
						var data = mfData.get(id);
						macBytes = new DataView(data.buffer.slice(0, 9));
						break;
					}
				}
			}

			if (!macBytes) {
				return Promise.reject('Cannot get MAC address');
			}

			var macParts = [];
			for (var i = 0; i < 6; i++) {
				macParts.push(('0' + macBytes.getUint8(i).toString(16)).slice(-2));
			}
			deviceMac = macParts.join(':').toUpperCase();
			giikerutil.log('[gan251cube] MAC:', deviceMac);

			var keyIv = deriveKeyIv(deviceMac);
			if (!keyIv) {
				return Promise.reject('Cannot derive key/IV');
			}

			decoder = $.aes128(keyIv.key);
			decoder.iv = keyIv.iv;
			decoder.key = keyIv.key;

			return _gatt.getPrimaryService(SERVICE_UUID_DATA);
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
		batteryLevel = 0;
		return result;
	}

	GiikerCube.regCubeModel({
		prefix: ['GAN251', 'gan251ui_', 'ganic251_', 'gan251ui'],
		init: init,
		opservs: [SERVICE_UUID_DATA],
		cics: GAN251_CIC_LIST,
		getBatteryLevel: getBatteryLevel,
		clear: clear
		
	});
});
