"use strict";

var remotecubeUtil = execMain(function() {

	var urlInput = $('<input type="text" style="width:100%">').val(localStorage['remoteCubeUrl'] || '');
	var statusDiv = $('<span>');
	var saveBtn = $('<input type="button" class="buttonOK" value="保存">');
	var serialBtn = $('<input type="button" class="buttonOK" value="连接串口">');
	var modeSpan = $('<span>');
	var solvedEnterCb = $('<input type="checkbox">');

	function execFunc(fdiv) {
		if (!fdiv) return;
		urlInput.val(localStorage['remoteCubeUrl'] || '');
		solvedEnterCb.prop('checked', localStorage['remoteCubeSolvedEnter'] == '1');
		refreshMode();
		fdiv.empty().append(
			$('<p>').text('远程魔方地址:'),
			urlInput,
			'<br><br>',
			saveBtn,
			'<br><br>',
			$('<p>').text('下发方式:'),
			modeSpan,
			serialBtn,
			'<br><br>',
			$('<label>').append(solvedEnterCb, '<span class="click"> 复原态可进入</span>'),
			'<br><br>',
			statusDiv
		);
		saveBtn.unbind('click').click(function() {
			$.waitUser.call();
			var url = urlInput.val().trim();
			localStorage['remoteCubeUrl'] = url;
			statusDiv.text('已保存').css('color', '#0f0');
			setTimeout(function() { statusDiv.text(''); }, 2000);
		});
		serialBtn.unbind('click').click(function() {
			$.waitUser.call();
			if (RemoteCube.getMode() == 'serial') {
				RemoteCube.setMode('http').then(refreshMode);
			} else {
				RemoteCube.setMode('serial').then(refreshMode, function(err) {
					statusDiv.text('串口连接失败: ' + err).css('color', '#f00');
				});
			}
		});
		solvedEnterCb.unbind('click').click(function() {
			$.waitUser.call();
			localStorage['remoteCubeSolvedEnter'] = solvedEnterCb.prop('checked') ? '1' : '0';
		});
	}

	function refreshMode() {
		var mode = RemoteCube.getMode();
		modeSpan.text('当前: ' + (mode == 'serial' ? '串口' : 'HTTP'));
		serialBtn.val(mode == 'serial' ? '切换到 HTTP' : '连接串口');
	}

	$(function() {
		tools.regTool('remotecube', TOOLS_REMOTECUBE, execFunc);
	});

	return {};
});
