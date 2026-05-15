# 转动序列实时坐标系转换

**Date:** 2026-05-15

## Context

用户需要在白顶绿前（WCA 标准）坐标系下解读魔方状态，但外部输入（如蓝牙魔方、GiiKER、手动操作）的转动序列中包含转体操作（y/x/z）、双层转动（Uw/Rw/Fw）和夹层转（M/E/S）。需要将这些操作实时转换为等效的原始坐标下的单层转动序列。

## Discussion

- **ori (0-23)**: 表示魔方的 24 种空间取向，对应立方体旋转群 SO(3) 的离散子群。0 为初始方向（U 上 F 前）。
- **rotMult[i][j]**: 旋转复合表（Cayley 表），`rotCube[i] ∘ rotCube[j] = rotCube[k]`, `rotMult[i][j] = k`。
- **rotMulI[k][j]**: 逆元索引表，`rotMulI[0][ori]` 是 `ori` 的逆旋转索引。
- **rotMulM[i][j]**: 共轭转动表，`rotMulM[i][j] = m` 表示旋转 i 将基本转动 j 共轭为 m，等价于 `rot * move * rot⁻¹ = move'`。
- **双层转动处理**: 拆解为对面单层转 + 转体（如 Uw = D + y）。
- **夹层转处理**: 拆解为两个对面单层转 + 转体（如 M = R + L' + x）。

## Approach

维护当前 ori（初始 0），遍历序列中的每个操作：

1. **转体操作** (y/x/z): 查找 `rot2str` 获取对应 rot 索引 r，更新 `ori = rotMult[r][ori]`。
2. **单层转动** (U/R/F/D/L/B): 将面名转 move 索引 m，用 `rotMulM[ori][m]` 得到原始坐标下的等效 move，再反解为面名。
3. **双层转动** (Uw/Rw/Fw/Dw/Lw/Bw): 拆为对面单层转 + 转体，分别按 1 和 2 处理。
4. **夹层转动** (M/E/S): 拆为两个对面单层转 + 转体，分别处理。

## Architecture

核心利用 mathlib 现有接口：

```
moveIndex = axis * 3 + (power % 4 - 1)        // 0-17
convertedIdx = CubieCube.rotMulM[ori][moveIndex]
convertedFace = "URFDLB"[Math.floor(convertedIdx / 3)]
convertedPower = convertedIdx % 3 + 1          // 1=单层, 2=180°, 3=逆
```

ori 更新逻辑（转体时）：

```
rotIndices = { y: 3, x: 15, z: 17 }
ori = CubieCube.rotMult[rotIndices[face]][ori]
```

现有 `selfMoveStr` 方法（`mathlib.js:787`）已实现了同样的转换逻辑，可直接复用或参考其实现。
