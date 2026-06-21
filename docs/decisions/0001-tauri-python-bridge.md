# ADR 0001: Tauri + Python Bridgeを採用する

日付: 2026-06-21

## 状態

採用

## 背景

LoRA学習GUIは、ローカルファイル、外部Python環境、GPU状態、長時間プロセス、画像プレビューを扱う。既存ツールにはブラウザGUI、WinForms GUI、Express UIがあるが、今回の目的では次の問題がある。

- 固定ポートやブラウザサーバーに依存したくない。
- sd-scripts級の詳細設定を扱いたい。
- マルチGPU、サンプル生成、TensorBoard、WanDBを一体で管理したい。
- Windows上でローカルツールとして扱いやすくしたい。

テンプレートにはTauriからPython moduleを呼ぶbridgeがあり、この方向が用途に合う。

## 決定

GUI本体はTauri + Vite + TypeScriptで作る。

Rust/Tauri commandはOS統合とプロセス制御を担当する。

Python bridgeは学習ドメイン処理を担当する。

外部学習エンジンはengine adapterとして扱い、アプリ本体に巨大な学習コードを持ち込まない。

## 理由

- Tauriなら固定HTTPポートなしでデスクトップGUIにできる。
- Python bridgeなら既存学習エンジン、TOML、画像処理、taggerと相性が良い。
- Rust側でprocess tree、ファイル選択、通知などのOS操作を安全に包める。
- TypeScript側はフォームと状態管理に集中できる。
- 既存ツールを丸ごと移植せず、能力だけを取り込める。

## 影響

良い影響:

- UIと学習実行の責務が分かれる。
- 複数エンジン対応を後から追加しやすい。
- ジョブ設定を構造化して保存できる。

悪い影響:

- Tauri/Rust/TypeScript/Pythonの複数技術スタックになる。
- bridgeの型定義とバージョン管理が必要。
- 長時間プロセスの停止、ログ転送、クラッシュ復旧を丁寧に作る必要がある。

## 代替案

### Express/Nextだけで作る

既存Standaloneに近いが、ポート管理とブラウザ起動が残る。デスクトップアプリとしてのOS統合も弱い。

### Python GUIで作る

学習ドメインとは近いが、UIの保守性、画像ブラウザ、複雑なフォーム、将来の見た目改善で不利。

### .NET/WinFormsを継承する

Kohya GUI資産とは近いが、今回求めるTauri方向、柔軟な画面構成、bridge設計には合わない。
