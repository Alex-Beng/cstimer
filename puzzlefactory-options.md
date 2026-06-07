# puzzlefactory 选项文档

## 概述

`src/js/lib/puzzlefactory.js` 是 csTimer 的魔方渲染工厂，接收两个核心参数 `puzzle` 和 `style`，分别控制**显示什么魔方**和**用什么方式渲染**。

---

## 一、`puzzle` 选项 — 定义魔方类型

`puzzle` 由 `options['puzzle']` 传入，决定魔方的几何体、维度、面颜色。最终 `options['type']` 被设置为实际注册的渲染类型。

### 完整列表

| puzzle 值 | options.type | 维度/说明 | 面颜色属性 |
|---|---|---|---|
| `cube3`, `cube4`, `cube5`... | `cube` | NxN 魔方，`dimension = N`，默认 3 | `colcube` (U L F D R B) |
| `skb` | `skewb` | Skewb | `colskb` |
| `mgm` | `mgm` | Megaminx | `colmgm` |
| `prc` | `prc` | Pyraminx Crystal | `colmgm` |
| `klm` | `klm` | Kilominx | `colmgm` |
| `giga` | `giga` | Gigaminx | `colmgm` |
| `pyr` | `pyr` | Pyraminx | `colpyr` |
| `mpyr` | `mpyr` | Master Pyraminx | `colpyr` |
| `sq1` | `sq1` | Square-1 | `colsq1` |
| `clk` | `clk` | Clock | `colclk` |
| `fto` | `fto` | Face-Turning Octahedron | `colfto` |
| `dmd` | `dmd` | Diamond | `colfto` |
| `heli` | `heli` | Helicopter Cube | `colcube` |
| `heli2x2` | `heli2x2` | 2x2 Helicopter | `colcube` |
| `helicv` | `helicv` | Curvy Copter | `colcube` |
| `crz3a` | `crz3a` | Crazy 3x3 | `colcube` |
| `redi` | `redi` | Redi Cube | `colcube` |
| `dino` | `dino` | Dino Cube | `colcube` |
| `ctico` | `ctico` | Icosamate (二十面体) | `colico` |
| `udpoly` | `udpoly` | 用户自定义多面体 | 根据面数自动选择 |

### puzzle 的值从何而来

用户选择打乱类型（如 `333`、`skbso`）后，通过 `tools.js` 中的 `puzzleType()` 函数映射为对应的 `puzzle` 值（如 `cube3`、`skb`）。

---

## 二、`style` 选项 — 定义渲染方式

`style` 由 `options['style']` 传入，决定使用哪个渲染引擎。核心解析逻辑（puzzlefactory.js 第 82 行）：

```js
var style = /^q[2l]?$/.exec(options['style']) ? 'q' : 'v';
```

### 有效值

| 值 | 实际解析 | 渲染引擎 | 说明 |
|---|---|---|---|
| `'v'` | `'v'` | **Three.js 3D** (`twistyjs.TwistyScene`) | 完整 WebGL 3D 渲染，支持所有 puzzle 类型 |
| `'q'` | `'q'` | **qCube 2D Canvas** (`qcube.TwistyScene`) | 2D 平面展开图渲染，仅支持 `cube`、`mgm`、`clk` |
| `'ql'` | `'q'` | qCube 2D Last Layer | 只显示顶层(U)和前面(F)，仅 NxN cube |
| `'q2'` | `'q'` | qCube 2D Two-Look | 显示 U/R/F 三面，仅 NxN cube |

### style 的值从何而来

- **输入模式** (`timer.input`)：属性值为 `['t','i','s','m','v','g','q','b','l','r']`，其中 `v` → 3D，`q` → 2D
- **智能魔方 VRC 模式** (`vrc.giiVRC`)：属性值为 `['n','v','q','ql','q2']`
  - `n` = 无显示
  - `v` = 3D 虚拟
  - `q` = qCube 完整
  - `ql` = qCube 最后一层
  - `q2` = qCube Two-Look

---

## 三、`puzzle` vs `style` 的区别

| 维度 | `puzzle` | `style` |
|---|---|---|
| **定义** | **画什么**（魔方几何体） | **怎么画**（渲染引擎/视图） |
| **作用** | 决定几何体、维度、面颜色分布 | 决定 3D / 2D / 局部视图 |
| **来源** | 由打乱类型 (`scrType`) 映射得到 | 由输入模式或 VRC 设置得到 |
| **影响范围** | 决定 `type`、`dimension`、`faceColors` | 决定使用 `twistyjs.TwistyScene` 还是 `qcube.TwistyScene` |
| **可取值** | 约 20+ 种 puzzle ID | `v`, `q`, `ql`, `q2`（四种） |

### 典型组合示例

| 场景 | puzzle | style | 结果 |
|---|---|---|---|
| 虚拟 3x3 3D | `cube3` | `v` | Three.js 渲染完整 3D 魔方 |
| qCube 3x3 2D | `cube3` | `q` | qCube 渲染 2D 展开图 |
| VRC Last Layer | `cube3` | `ql` | qCube 只显示 U 面和 F 面 |
| 虚拟 Pyraminx | `pyr` | `v` | Three.js 渲染 3D 金字塔 |
| qCube Megaminx | `mgm` | `q` | qCube 渲染 2D 五魔方（忽略 ql/q2） |

### 渲染引擎的局限性

- **Three.js (v)**：支持所有 puzzle 类型
- **qCube (q/ql/q2)**：仅注册了 `cube`（qcubennn.js）、`mgm`（qcubeminx.js）、`clk`（qcubeclk.js）三种类型。其他 puzzle 在 qCube 下无法渲染
- `ql` 和 `q2` 仅在 NxN cube (`cube3` 等) 下有效；`mgm` 和 `clk` 忽略子样式，始终显示完整视图

---

## 四、相关源码文件

| 文件 | 作用 |
|---|---|
| `src/js/lib/puzzlefactory.js` | 核心工厂，解析 puzzle/style 并初始化渲染引擎 |
| `src/js/twisty/twisty.js` | Three.js 3D 渲染引擎（`twistyjs.TwistyScene`） |
| `src/js/twisty/qcube.js` | 2D Canvas 渲染引擎（`qcube.TwistyScene`） |
| `src/js/twisty/twistynnn.js` | 注册 3D NxN cube 渲染 |
| `src/js/twisty/twistypoly.js` | 注册 3D 多面体（mgm, pyr, fto 等）渲染 |
| `src/js/twisty/twistysq1.js` | 注册 3D Square-1 渲染 |
| `src/js/twisty/twistyskb.js` | 注册 3D Skewb 渲染 |
| `src/js/twisty/twistyclk.js` | 注册 3D Clock 渲染 |
| `src/js/twisty/qcubennn.js` | 注册 qCube NxN cube + 处理 q/ql/q2 子样式 |
| `src/js/twisty/qcubeminx.js` | 注册 qCube Megaminx |
| `src/js/twisty/qcubeclk.js` | 注册 qCube Clock |
| `src/js/lib/utillib.js` | 定义 `$.TWISTY_RE` / `$.UDPOLY_RE` 正则 |
| `src/js/tools/tools.js` | `puzzleType()` 映射 scrType → puzzle 值 |
