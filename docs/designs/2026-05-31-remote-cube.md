# 远程智能魔方 (Remote Cube) 技术文档

**日期:** 2026-05-31  
**目标:** 通过 HTTP/串口 将远程智能魔方接入 cstimer，实现远程示教功能

---

## 1. 架构概览

```
┌─────────────────────────────────────────────┐
│                  cstimer                     │
│  ┌──────────┐    ┌────────────────────┐     │
│  │ tool     │    │ timer/remotecube.js │     │
│  │ remotecube│◄──│    (计时器集成)     │     │
│  │  (设置)   │    │  moveListener      │     │
│  └──────────┘    │  processMoveForAPI │     │
│                   │  scrambleIt         │     │
│                   └────────┬───────────┘     │
│                            │                  │
│                   ┌────────▼───────────┐     │
│                   │hardware/remotecube │     │
│                   │    (传输层)         │     │
│                   │  HTTP / 串口        │     │
│                   └────────┬───────────┘     │
└─────────────────────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   远程智能魔方     │
                    │  REST API:        │
                    │  GET /api/facelets│
                    │  POST /api/moves  │
                    └───────────────────┘
```

## 2. 新增/修改文件清单

| 文件 | 说明 |
|------|------|
| `src/js/hardware/remotecube.js` | HTTP 客户端 + 串口传输，含发送队列 |
| `src/js/timer/remotecube.js` | 计时器集成模块，复刻 virtual.js 流程 |
| `src/js/tools/remotecube.js` | 工具页面：URL/模式/开关设置 |
| `src/index.php` | 加载上述 3 个脚本 |
| `src/js/timer.js` | 注册 `input='r'` 路由 + LCD 适配 |
| `src/js/shortcut.js` | Ctrl+Alt+R 快捷键 |
| `Makefile` | 加入编译目标 |
| `src/lang/*.js` (28 个) | `PROPERTY_ENTERING_STR` 加入"远程魔方"，`TOOLS_REMOTECUBE` 标签 |

## 3. 工作流程

```
1. Ctrl+Alt+R 或下拉菜单 → input='r'
2. 首次弹出 URL 对话框 → 存入 localStorage
3. 初始化 3D puzzle (style='v') 到还原状态
4. 按空格 → GET /api/facelets → 同步远程状态 → 观察/还原
5. 转动 → processMoveForAPI → 坐标转换 → 排队发送
6. 还原 → pushSignal('time') → 记录成绩
7. 按空格 → 重复步骤 4
```

## 4. 坐标系转换

### 4.1 核心概念

远程魔方使用**白顶绿前**（WCA 标准）固定坐标系 (`ori=0`)。用户键盘操作中可能包含转体 (`x/y/z`)，使当前展示视角偏离原始坐标。所有下发转动必须转换回原始坐标系。

维护一个 **ori**（0-23），表示当前立方体空间取向，每次转体操作更新它，单层/双层/夹层转动用它做共轭变换。

### 4.2 ori 的定义与运算

24 种旋转对应立方体旋转群 SO(3) 的离散子群，编码为 0-23。**所有表由 mathlib 在启动时自动生成**，无需手动推导。

```js
// mathlib.js:680-727 自动生成
CubieCube.rotCube   // 24 个旋转的 CubieCube
CubieCube.rotMult   // Cayley 表: 旋转复合 rot_i ∘ rot_j
CubieCube.rotMulI   // 逆元表
CubieCube.rotMulM   // 共轭表: rot⁻¹ * move * rot
CubieCube.rot2str   // rot 编号 → 字符串 "y", "x' z" 等
```

使用时直接用 `mathlib.CubieCube.rotMulM[ori][moveIdx]` 查表即可。

### 4.3 moveIdx 编码

18 个基础转动编号：6 个面 × 3 种力度 = 0-17。

| axis | moveIdx | 转动 |
|------|---------|------|
| U(0) | 0,1,2 | U, U2, U' |
| R(1) | 3,4,5 | R, R2, R' |
| F(2) | 6,7,8 | F, F2, F' |
| D(3) | 9,10,11 | D, D2, D' |
| L(4) | 12,13,14 | L, L2, L' |
| B(5) | 15,16,17 | B, B2, B' |

编码：`moveIdx = axis × 3 + pow - 1`，其中 `pow = 1(CW)、2(双)、3(CCW)`。  
解码：`axis = Math.floor(idx/3)`, `pow = idx % 3`。

### 4.4 分类型处理

以下 `processMoveForAPI` 的完整逻辑（`timer/remotecube.js:52-97`）：

#### 单层转动 (URFDLB)

```js
var axis = 'URFDLB'.indexOf(face);   // 0=U,1=R,2=F,3=D,4=L,5=B
var pow = "2'".indexOf(m[2] || '-') + 2; // 1=CW,2=双,3=CCW
var moveIdx = axis * 3 + pow % 4 - 1;
var convertedIdx = rotMulM[ori][moveIdx];  // 共轭变换
decodeMoveIdx(convertedIdx);               // 解码后入队下发
```

**示例：** 当前 `ori=3`（已做 `y` 转体），用户按 `H` 做 `F` 转动：

```
axis = 2 (F)
moveIdx = 2*3 + 1-1 = 6 (F)
convertedIdx = rotMulM[3][6] → 12 (L)  // y*F*y⁻¹ = L
下发 {face: 4, dir: 0}                    // L = face 4
```

实际效果：用户视角看是转 F 面，但远程魔方用固定坐标系该转 L 面。

#### 双层转动 (UwRwFwDwLwBw)

宽转分解为对面单层 + 转体。以 Rw 为例：`Rw = L + x`。

```js
axis >>= 1;                                   // 还原到 0-5 面索引
var moveIdx = (axis + 3) % 6 * 3 + pow%4 - 1; // 对面单层
decodeMoveIdx(rotMulM[ori][moveIdx]);          // 对面单层转换后下发
var rot = [3, 15, 17, 1, 11, 23][axis];       // y,x,z,y',x',z'
for (var i = 0; i < pow; i++) {
    ori = rotMult[rot][ori];                   // 更新朝向
}
```

**示例：** `ori=0` 时做 `Rw`（I+3 双层键）：

```
axis = 1 (R), pow = 1
对面 = (1+3)%6 = 4 → L
moveIdx = 4*3 + 1-1 = 12 (L)
convertedIdx = rotMulM[0][12] = 12 → 下发 {face:4, dir:0}
rot = [3,15,17,1,11,23][1] = 15 (x)
ori = rotMult[15][0] = 15
```

下发 `L`（面 4），本地更新 `ori=15` 表示 x 转体发生。

#### 转体 (xyz)

纯转体不下发转动，只更新 ori。

```js
var rot = [3, 15, 17][axis];  // y=3, x=15, z=17
for (var i = 0; i < pow; i++) {
    ori = rotMult[rot][ori];
}
```

**示例：** `ori=0` 时做 `y`（；键）：

```
rot = 3 → ori = rotMult[3][0] = 3
```

后续所有转动都用 `rotMulM[3][*]` 转换。

#### 夹层转动 (MES / 2-2Xw)

夹层转分解为两个对面单层 + 转体。以 `M` 为例：`M = L' + R + x'`。

```js
var m1 = axis*3 + (4-pow)%4 - 1;       // 一侧反向
var m2 = (axis+3)%6*3 + pow%4 - 1;     // 对侧正向
decodeMoveIdx(rotMulM[ori][m1]);        // 下发两个单层
decodeMoveIdx(rotMulM[ori][m2]);
var rot = [3,15,17,1,11,23][axis];      // 更新 ori
for (var i = 0; i < pow; i++) {
    ori = rotMult[rot][ori];
}
```

### 4.5 decodeMoveIdx → API 格式

```js
function decodeMoveIdx(idx) {
    var face = Math.floor(idx / 3);  // 0=U,1=R,2=F,3=D,4=L,5=B → 直接等于 API face
    var pow = idx % 3;               // 0=CW, 1=双, 2=CCW
    if (pow == 1) {
        enqueueMove(face, 0);        // 双转为两次 CW
        enqueueMove(face, 0);
    } else {
        enqueueMove(face, pow == 2 ? 1 : 0);  // CW→0, CCW→1
    }
}
```

API face 顺序（0=U,1=R,2=F,3=D,4=L,5=B）与 `rotMulM` 解码出的 axis 完全一致，无需额外映射。

### 4.6 ori 生命周期

```
┌──────────┐   空格/连接    ┌──────────┐
│ resetOri │──────────────→│  ori = 0  │
│ (ori=0)  │               │  (初始)   │
└──────────┘               └─────┬─────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                转体 xyz     双层 Rw/Uw    夹层 MES
                    │        ┌───┴───┐    ┌──┴──┐
                    ▼        ▼       ▼    ▼     ▼
              rotMult 更新   下发对面  更新   下发两个  更新
              ori          单层     ori   单层    ori
```

- 每次 `scrambleIt()`（按空格）都调用 `resetOri()` 归零
- 观察期和还原期的所有转体操作持续更新 `ori`
- `ori` 只在本地跟踪，不下发给远程魔方

## 5. 传输方式

### HTTP 模式（默认）
- `GET /api/facelets` → 返回 kociemba 54字符面片串
- `POST /api/moves` → body: `[{"face":0-5,"dir":0-1},...]`
- **串行队列**：快速操作时多个请求排队串行发送，防止乱序到达
- **CORS**：需服务端返回 `Access-Control-Allow-Origin: *`

### 串口模式
- 基于 Web Serial API（仅 Chrome/Edge，需 HTTPS/localhost）
- 数据格式：`[{"face":0,"dir":0},...]\n` 写入串口
- 波特率：115200

### 切换方式
- 工具页面"连接串口"按钮切换，状态存 `localStorage['remoteCubeMode']`

## 6. 工具页面开关

| 开关 | localStorage key | 默认 | 说明 |
|------|------------------|------|------|
| 复原态可否进入 | `remoteCubeSolvedEnter` | 关 | 开时复原态也进入还原流程 |
| 开启下发 | `remoteCubeSendEnable` | 开 | 关时本地操作正常但不下发转动 |

复原态逻辑：
- 未勾选 + 下发给开 → 发送 R R' 后返回
- 未勾选 + 下发已关 → 静默返回
- 已勾选 → 正常进入还原流程

## 7. 成绩产生流程

完全复刻 `virtual.js` 的观察→还原→计时→成绩产生：

```
status: -1(空闲) → -3(观察) → -2(就绪) → ≥1(还原阶段) → solved → pushSignal('time')
```

- 观察期转体记录到 `rawMoves[0]`，`processMoveForAPI` 同步更新 `ori`
- 首个非转体转动开始计时，进入还原阶段
- 还原阶段所有转动记录 + `processMoveForAPI` 转换下发
- `mstep==2` 且 `curProgress==0` → 还原完成，记录成绩
- ESC → 取消/DNF

成绩 `pushSignal` 格式：
```js
['time', ["", 0, curTime, 0, [reconStr, '333', moveCnt]]]
```

同时 `doScramble` 中 push 打乱信号保证回放正确：
```js
kernel.pushSignal('scramble', ['333', cubeutil.getConjMoves(gen, true), 0]);
```

## 8. 已知限制

- HTTP 需 CORS 头或同源部署；HTTPS 页面禁止请求 HTTP 资源
- Web Serial 仅 Chrome/Edge 支持
- 远程魔方必须在同一网络可达
