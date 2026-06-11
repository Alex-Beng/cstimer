# csTimer — Agent 指南

## 构建与开发命令

- **根目录无 `package.json`** — 构建系统基于 `Makefile`
- 构建依赖：**Java**（通过 `lib/compiler.jar` 运行 Closure Compiler）和 **PHP**
- `make all` — 构建 `dist/`（JS、CSS、语言文件、缓存清单、Service Worker）
- `make local` — 构建静态离线快照到 `dist/local/`
- `make module` — 构建 `npm_export/cstimer_module.js`（npm 包）
- `make check` — 仅运行 Closure Compiler 检查模式（无输出）
- `make clean` — 删除构建产物
- 输出目录 `dist/` 已被 `.gitignore` 忽略

## 架构

- **纯 JS 单页应用** — jQuery 1.8.0，无打包器，无框架
- **无 TypeScript、无 ESLint、无 Prettier、无格式化工具** — 仅原生 JS
- 开发模式：`src/index.php` 直接加载约 100 个独立 `<script>` 标签
- 生产模式：Closure Compiler 合并压缩为 `dist/js/cstimer.js` 和 `dist/js/twisty.js`
- 核心：`src/js/kernel.js` 中的信号/发布订阅事件系统
- 打乱生成在 **Web Worker** 中运行（`src/js/worker.js` + `src/js/cstimer.js`）
- 所有用户数据仅存储在客户端（localStorage + IndexedDB）

## npm 模块

- `npm_export/` 拥有独立的 `package.json`（以 `cstimer_module` 发布在 npm）
- 测试：`cd npm_export && npm test` — 运行 `testbench/test.js`（基础测试 + 基准测试）
- 类型声明：`npm_export/cstimer_module.d.ts`

## 关键源码布局

| 目录 | 用途 |
|------|------|
| `src/js/` | 所有 JS 源码（开发模式） |
| `src/js/lib/` | 核心库（数学、群论、min2phase 等） |
| `src/js/scramble/` | 各魔方类型的打乱生成器 |
| `src/js/twisty/` | 3D 魔方可视化（基于 Three.js） |
| `src/js/tools/` | 工具（十字求解器、盲拧助手等） |
| `src/js/hardware/` | 蓝牙/智能魔方驱动 |
| `src/js/stats/` | 统计、趋势、分布 |
| `src/lang/` | 国际化（73 种语言：`.js` 字符串 + `.php` 模板） |
| `lib/` | 构建时依赖：`compiler.jar`、`jquery-1.7.js` |
| `experiment/` | 实验性文件（不参与构建） |
| `docs/designs/` | 设计文档 |

## CI

- `.github/workflows/pages.yml` — 使用 Java+PHP 构建 `master` 和 `moyu` 分支，部署到 GitHub Pages
- 推送到 `master` 或 `moyu` 分支时自动触发部署

## `moyu` 分支 vs `master`（当前分支）

`moyu` 是一个面向智能硬件和功能迭代的特性分支，相比 `master` 已超前约 90+ commits。核心差异：

| 领域 | 变更 |
|------|------|
| **新硬件支持** | `src/js/hardware/gan251cube.js` — GAN 251 智能魔方驱动（519行） |
| **远程魔方** | 新增 `src/js/hardware/remotecube.js`、`src/js/timer/remotecube.js`、`src/js/tools/remotecube.js` — 通过 HTTP/串口远程操作实体魔方，支持 2 阶和 3 阶 |
| **2 阶智能魔方** | `src/js/timer/giiker.js` 大幅重写 — 2x2 无棱块的特殊处理（`isSolvedState`、`checkScramble`、`markSolved`），打乱校验仅比较角块 |
| **蓝牙层** | `src/js/hardware/bluetooth.js` — 蓝牙前缀优先级调整（GAN251 优先）|
| **打乱生成** | `src/js/scramble/2x2x2.js` — 2x2 远程打乱适配 |
| **工具入口** | `src/js/tools/bluetoothutil.js` — 新增对 `222` 魔方类型的 UI 支持 |
| **设计文档** | `docs/designs/2026-05-15-rotation-coordinate-transform.md` — 旋转坐标系变换设计 |
| **国际化** | 全部 36 个语言文件统一新增 `TOOLS_REMOTECUBE` 条目和 `remote cube` 输入方式 |
| **构建相关** | `.gitignore` 忽略更多生成文件，`Makefile` 小幅调整，`src/index.php` 新增脚本引用 |

### 开发注意事项

- **2x2 智能魔方调试日志丰富** — `giikerutil.log(...)` 遍布整个 `giiker.js`，调试时可搜索 `[btutil-cb]`、`[chkScr]` 等前缀
- **远程魔方有两种通信模式** — HTTP（POST 请求）和 Web Serial API（串口），URL 保存在 `localStorage['remoteCubeUrl']` 中
- **GAN 251 使用 Gen4 协议**，与老款 GAN 魔方协议不兼容（见 `gan251cube.js` 与 `gancube.js` 的区分）