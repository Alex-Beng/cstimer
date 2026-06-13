# Plan: GAN251 Merge + 2x2 Special-Case Refactor

## Background

From firmware emulator (`gan-cube-emu/src/protocol.rs`), **Gen4 and Gan251 decrypted packet formats are identical**:

- **Move packet**: `byte[0]=0x01`, `bytes[2..5]=timestamp LE`, `bytes[6..7]=serial LE`, `byte[8]=(dir<<6)|face_bitmap`
- **State packet**: `byte[0]=0xED`, `bytes[2..3]=serial LE`, `byte[4+]=payload`
  - Payload differs slightly: Gen4 uses 7 CO + 2 pad + 11 EO; Gan251 uses 8 CO + 12 EO
  - V4's `parseV4Data` is compatible with both (uses checksum inference for 8th/12th)

Only **real differences**:
1. Key derivation base constants (`GAN251_BASE_KEY/IV` vs `KEYS[2..5]`)
2. `puzzleSize`: 2 vs 3
3. `byte[1]` length field: 7 (Gan251) vs 9 (Gen4) in move; 14 vs 12 in state — parser ignores this

## Part 1: Merge GAN251 into `gancube.js`

### Changes to `src/js/hardware/gancube.js`

1. **Add GAN251 constants** (from `gan251cube.js`):
   - `GAN251_BASE_KEY[16]`, `GAN251_BASE_IV[16]`
   - `deriveKeyIv(mac)` function

2. **Modify `init()`**:
   - After `device.gatt.connect()`, detect GAN251 by `device.name` prefix
   - If GAN251: extract MAC from manufacturer data, derive key via `deriveKeyIv()`, then call `gan251Init()`
   - `gan251Init()`: same V4 service setup + `onStateChangedV4` handler + requests (hardware/facelets/battery)

3. **Add second `GiikerCube.regCubeModel()`**:
   ```javascript
   GiikerCube.regCubeModel({
       prefix: ['GAN251', 'gan251ui_', 'ganic251_', 'gan251ui'],
       init: init,
       opservs: [SERVICE_UUID_V4DATA],
       cics: GAN_CIC_LIST,
       getBatteryLevel: getBatteryLevel,
       clear: clear,
       puzzleSize: 2
   });
   ```

4. **`gan251Init()`** differs from `v4init()` only in key derivation:
   - No `v2initKey()` — uses `deriveKeyIv()` instead
   - Everything else (service setup, requests) is shared

5. **`clear()`** — already handles V4 cleanup, no changes needed

### Remove `src/js/hardware/gan251cube.js`

### Update references
- `src/index.php` line 91: remove `<script src="js/hardware/gan251cube.js">`
- `Makefile` line 93: remove `hardware/gan251cube.js \`

## Part 2: Centralize 2x2 Special Cases

### Add helpers to `src/js/lib/cubeutil.js`

```javascript
function is222Solved(facelet) {
    var idx = [0, 2, 6, 8];
    for (var b = 0; b < 54; b += 9)
        for (var i = 0; i < 4; i++)
            if (facelet[b + idx[i]] != mathlib.SOLVED_FACELET[b + idx[i]])
                return false;
    return true;
}

function can222Start(facelet) {
    var cc = new mathlib.CubieCube();
    cc.fromFacelet(facelet);
    for (var i = 0; i < 8; i++)
        if (cc.ca[i] != i) return true;
    return kernel.getProp('giiMode') != 'n';
}

function getCurScrambler() {
    return tools.getCurPuzzle() == '222' ? scramble_222 : scramble_333;
}
```

### Update `src/js/timer/giiker.js`

| Location | Change |
|----------|--------|
| `canStart()` line 248 | Replace `tools.getCurPuzzle() == '222'` block → `cubeutil.can222Start(facelet)` |
| `isGiiSolved()` line 278 | Replace inner 2x2 corner check → `cubeutil.is222Solved(facelet)` |
| `markScrambled()` line 297 | `scramble_222/scramble_333` ternary → `cubeutil.getCurScrambler().genFacelet(...)` |
| VRC `setState()` line 96 | Same ternary → `cubeutil.getCurScrambler().genFacelet(...)` |

### Update `src/js/tools/bluetoothutil.js`

| Location | Change |
|----------|--------|
| `isSolvedState()` line 372 | Replace facelet-inner 2x2 loop → `cubeutil.is222Solved(facelet)` |
| `checkState()` line 98 | `cubeModel.puzzleSize == 2 ? scramble_222 : scramble_333` → `cubeutil.getCurScrambler()` |
| `checkScramble()` line 135 | Keep as-is (different logic — compares scramble state to current state) |
| `markSolved()` line 393 | Keep as-is (different logic — CubieCube multiplication) |

## Part 3: Fix "Reset bluetooth cube as solved?" Cancel → Still Shows Solved

### Root Cause

`bluetoothutil.init()` loads `solvedStateInv` from `giiSolved` (saved by a previous `markSolved()`):
```javascript
curRawState = kernel.getProp('giiSolved', mathlib.SOLVED_FACELET);
curRawCubie.fromFacelet(curRawState);
solvedStateInv.invFrom(curRawCubie);  // ← non-identity
```

When V4 facelet arrives, `initCubeState()` calls `GiikerCube.callback()` which transforms:
```javascript
CubieCube.CubeMult(solvedStateInv, curRawCubie, curCubie);
// curCubie = inv(oldState) * newState → relative state
```

If the physical cube hasn't moved since `markSolved()` was last called, `curCubie` is solved.
The confirm dialog (`rst == 'p'`) appears because `latestFacelet != giiSolved` (raw state ≠ saved).
But if user clicks Cancel:
- `markSolved()` is NOT called (correct)
- `solvedStateInv` remains non-identity → subsequent state tracking still shows wrong relative state

### Fix

1. **`bluetoothutil.js`**: Export `resetSolvedInv()` that resets `solvedStateInv = new CubieCube()` (identity)
2. **`gancube.js` `initCubeState()`**: On Cancel (`rst == 'p'` + confirm false), call `resetSolvedInv()` + re-run callback with raw facelet
3. Same fix applied to `qiyicube.js` and `moyu32cube.js` (same pattern)

## File Summary

| File | Lines Δ |
|------|---------|
| `gancube.js` | ~+130 lines |
| `gan251cube.js` | -519 lines (deleted) |
| `cubeutil.js` | ~+30 lines |
| `giiker.js` | ~-20 lines |
| `bluetoothutil.js` | ~-10 lines (Part 2) + ~+4 lines (Part 3) |
| `index.php` | -1 line |
| `Makefile` | -1 line |
| `qiyicube.js` | ~+3 lines |
| `moyu32cube.js` | ~+3 lines |
