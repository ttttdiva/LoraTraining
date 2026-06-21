# 既存ツール調査

調査日: 2026-06-21

対象:

- `D:\tool\lora_trainer\Anima-LoRA-Factory`
- `D:\tool\lora_trainer\Anima-Standalone-Trainer`
- `D:\tool\lora_trainer\Kohya_lora_param_gui`
- テンプレート: `D:\Dev\00_Tool_Template`

## 要約

このプロジェクトでは既存ツールを直接合体させず、次のように分解して使う。

- LoRA Factoryは「初心者導線、環境自動構築、タグ編集UX」の参照にする。
- Kohya lora param GUIは「設定項目、optimizer、sd-scripts互換コマンド」の参照にする。
- Anima Standalone Trainerは「ジョブ管理、TOML構成、マルチGPU、サンプル生成」の実行基盤参照にする。
- 画面はTauriで新規実装し、Webサーバー常駐UIや固定ポート依存は避ける。

## Anima-LoRA-Factory

確認できたローカル構成:

- README、ライセンス、画像、配布zipが中心。
- README上では `start.bat` からvenvを作成し、ブラウザGUIを開く方式。
- 特徴として、auto setup、visual tag editor、WD14 tagger、real-time progress、auto shutdown、ComfyUI conversionが挙げられている。

採るもの:

- 初回セットアップをユーザーに意識させない導線。
- 学習前処理、タグ付け、キャプション編集を学習ジョブ作成と一体化する流れ。
- GPUやPyTorch環境のチェックをGUI上で説明する作り。

避けるもの:

- GUIのためだけに固定ポートを占有する構造。
- 設定項目を狭くしすぎること。
- サンプル生成やマルチGPU設定をGUI外に逃がすこと。

## Kohya_lora_param_gui

確認できたローカル構成:

- .NET WinFormsアプリ。
- `TrainParams.cs` に多数の学習パラメータがまとまっている。
- `MyUtils.GenerateCommands()` が `accelerate launch` とsd-scripts向け引数を生成する。
- Dataset toolsにはタグ整形、タグ移動、タグ削除、画像とtxtの整合性チェック、低解像度画像の退避などが含まれる。

参考になる主な設定群:

- Optimizer: `AdamW8bit`, `AdamW`, `Lion`, `Lion8bit`, `DAdaptLion`, `prodigy`, `AdamWScheduleFree`, `RAdamScheduleFree`, `Came`, `Custom` など。
- Scheduler: `cosine_with_restarts`, `cosine`, `linear`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `cosine_with_min_lr`, `warmup_stable_decay`。
- Network: LoRA, LyCORIS, DyLoRA, LoRAFA, LoHA, LoKr。
- Dataset: bucket、caption extension、caption dropout、tag dropout、shuffle caption、keep tokens。
- DiT/Anima系: flow shift、timestep sampling、blocks to swap、Qwen3 path、VAE cache制御など。

採るもの:

- sd-scripts互換の広いパラメータ体系。
- custom optimizer名とoptimizer_argsを明示できる設計。
- コマンド生成前に危険なカスタムコマンドを制限する考え方。
- Dataset toolsをGUIの「前処理」タブとして再設計する材料。

避けるもの:

- WinForms前提の密集UIをそのまま移植すること。
- 初回導入とsd-scripts配置をユーザー判断に任せすぎること。
- 生成コマンド文字列だけを状態の本体にすること。

## Anima-Standalone-Trainer

確認できたローカル構成:

- Python学習スクリプト群。
- `training-ui` はExpress + WebSocket + TOMLで構成。
- `architectures.json` でAnima/Luminaのモデルパス、学習スクリプト、生成スクリプトを定義。
- ジョブごとに `config.toml`, `dataset.toml`, `sample_prompts.txt`, `output`, `logs` を持つ。

特に参考になる実装:

- `buildTrainingConfig()` でジョブ設定とグローバルモデルパスをマージし、実行用 `_merged_config.toml` を作る。
- `buildLaunchConfig()` でGPU選択、DDP、FSDP、FSDP2、DeepSpeed、TP/SPを分岐する。
- 学習前にpersistent generation serverを止めてVRAMを空ける。
- サンプル生成はone-shotとkeep-loadedの両方を持つ。
- GPU状態は `nvidia-smi` から取得し、学習中/生成中のGPUを表示する。
- TensorBoardはジョブごとに起動し、ポートはサービス用途に応じて扱う。

採るもの:

- ジョブ単位のTOML保存。
- 実行直前にグローバルパスとジョブ設定をマージする方式。
- マルチGPUモードの設計。
- サンプル生成と画像ブラウザの統合。
- GPUモニタリング。

避けるもの:

- Expressサーバー常駐をGUI本体にすること。
- JS側で巨大な画面ロジックを一枚に積むこと。
- コマンドを文字列連結だけで組み立てること。

## 00_Tool_Template

確認できた構成:

- `packages/desktop`: Tauri + Vite。
- Rust側にPython bridge実行コマンドがある。
- `packages/shared`, `packages/web`, `packages/mobile` も含む。

採るもの:

- TauriからPython moduleを呼ぶbridge構造。
- `desktop.config.json` 的なローカル設定。
- `pnpm` workspaceの考え方。

変えるもの:

- このプロジェクトでは初期段階でmobile/webを持たない。
- 文字化けしたテンプレート文書はコピーしない。
- Python bridgeは長時間ジョブ向けに非同期プロセス管理を前提に拡張する。
