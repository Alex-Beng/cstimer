"use strict";

var remotecubeUtil = execMain(function() {

	var urlInput = $('<input type="text" style="width:100%">').val(localStorage['remoteCubeUrl'] || '');
	var statusDiv = $('<span>');
	var saveBtn = $('<input type="button" class="buttonOK" value="保存">');

	function execFunc(fdiv) {
		if (!fdiv) return;
		urlInput.val(localStorage['remoteCubeUrl'] || '');
		fdiv.empty().append(
			$('<p>').text('远程魔方地址:'),
			urlInput,
			'<br><br>',
			saveBtn,
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
	}

	$(function() {
		tools.regTool('remotecube', TOOLS_REMOTECUBE, execFunc);
	});

	return {};
});
