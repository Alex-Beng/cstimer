"use strict";

var preferinfer = execMain(function() {
    var puzzle2model = {}; 
    
    function loadModel(puzzle) {
        return new Promise(function(resolve, reject) {   
            if (!(puzzle in puzzle2model)) {
                // for now, only support clk
                const model_path = 'model/{puzzle}.onnx'.replace("{puzzle}", puzzle);
                ort.InferenceSession.create(model_path)
                    .then(session => {
                        console.log("Loaded {puzzle} model".replace("{puzzle}", puzzle));
                        puzzle2model[puzzle] = session;
                        resolve(session);
                    })
                    .catch(error => {
                        console.log("Error loading model:", error);
                        reject(error);
                    });
                
            } else {
                resolve(puzzle2model[puzzle]);
            }
        });
    }

    // copied from image.js
    function clkScr2Status(moveseq) {
        console.log("moveseq: ", moveseq);
        var movere = /([UD][RL]|ALL|[UDRLy]|all)(?:(\d[+-]?)|\((\d[+-]?),(\d[+-]?)\))?/
        var movestr = ['UR', 'DR', 'DL', 'UL', 'U', 'R', 'D', 'L', 'ALL']
        var moves = moveseq.split(/\s+/);
        var moveArr = clock.moveArr;
        var flip = 9;
        var buttons = [0, 0, 0, 0];
        var clks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (var i = 0; i < moves.length; i++) {
            var m = movere.exec(moves[i]);
            if (!m) {
                continue;
            }
            if (m[0] == 'y2') {
                flip = 9 - flip;
                continue;
            }
            var axis = movestr.indexOf(m[1]) + flip;
            if (m[2] == undefined && m[3] == undefined) {
                buttons[axis % 9] = 1;
                continue;
            }
            var power;
            var actions = [];
            if (m[1] == 'all') {
                power = ~~m[2][0] * (m[2][1] == '+' ? -1 : 1) + 12;
                actions.push(8 + 9 - flip, power);
            } else if (m[2]) {
                power = ~~m[2][0] * (m[2][1] == '+' ? 1 : -1) + 12;
                actions.push(axis, power);
            } else {
                power = ~~m[3][0] * (m[3][1] == '+' ? 1 : -1) + 12;
                actions.push(axis, power);
                power = ~~m[4][0] * (m[4][1] == '+' ? -1 : 1) + 12;
                axis = (10 - axis % 9) % 4 + 4 + 9 - flip;
                actions.push(axis, power);
            }
            for (var k = 0; k < actions.length; k += 2) {
                for (var j = 0; j < 14; j++) {
                    clks[j] = (clks[j] + moveArr[actions[k]][j] * actions[k + 1]) % 12;
                }
            }
        }
        return clks;
    }

    function execFunc(fdiv) {
        if (!fdiv) {
            return;
        }
        var model;
        var status;
        var curScramble = tools.getCurScramble();
        var puzzleType = curScramble[0];
        if (puzzleType == 'input') {
            puzzleType = tools.scrambleType(curScramble[1]);
        }
        puzzleType = tools.puzzleType(puzzleType);
    
        if (puzzleType == "clk") {
            status = clkScr2Status(curScramble[1]);
        }
        
        if (typeof status !== "undefined") {
            loadModel(puzzleType).then(async session => {
                try {
                    const data = Float32Array.from(status);
                    const tensor = new ort.Tensor('float32', data, [1, status.length]);
                    const feeds = { input: tensor};
                    const results = await session.run(feeds);
                    const value = results.output.data
                    fdiv.html("Prefer Value: {v}".replace("{v}", value));
                } catch (e) {
                    fdiv.html("infer fail: {e}".replace("{e}", e));
                }
            }).catch(error => {
                fdiv.html("Error in load models, refresh please...")
                return;
            });
        }
        else {
            // TODO: localized
            fdiv.html("Not suported puzzle type...")
        }
    }

    $(function() {
        // load onnx model
        loadModel("clk").then(session => {
            console.log("Loaded clk model sucess");
        }).catch(error => {
            console.log("Fail to load clk model");
        });
        
        tools.regTool('preferinfer', TOOLS_PREDICT_PREFER, execFunc);
    });
});