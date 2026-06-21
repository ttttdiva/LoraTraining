# LoraTraining

Tauri + TypeScript + Python bridge で作る LoRA 学習GUIです。`D:\tool\lora_trainer` 配下の既存ツールを外部エンジンとして扱い、データセット整備、タグ編集、ジョブ設定、TOML生成、学習・サンプル生成・TensorBoard起動を1つのデスクトップアプリにまとめます。

## 実装済み

- Dashboard: Python / Node.js / pnpm / Cargo / Git / GPU / 既存エンジンの検出
- Dataset Studio: 画像とcaptionのスキャン、個別caption保存、欠損/空/孤立caption検出、低解像度検出
- Bulk Tag Edit: タグ追加、削除、置換、先頭移動、shuffle
- Jobs: ジョブ作成、編集、保存、複製、削除、`dataset.toml` と `_merged_config.toml` 生成
- Training Launch: Anima Standalone向け `accelerate launch` プラン生成、マルチGPUモード、WanDB環境変数、プロセス開始/停止、ログ表示
- Samples: `anima_gen.py` 用サンプル生成プラン、プロセス開始/停止、ログ表示
- TensorBoard: 空きポート検出、起動、URL表示
- WD14 Tagger: モデル状態確認、依存導入プラン、モデルダウンロード、データセットへの自動タグ付け

## モデル配置

WD14 Taggerの既定配置先:

```text
D:\AI\models\Hot\image\Tagger\
```

このディレクトリに以下を配置します。

- `model.onnx`
- `selected_tags.csv`

GUIの Tagger 画面からもダウンロードできます。

## 開発コマンド

```powershell
pnpm install
pnpm typecheck
pnpm --filter desktop build
pnpm --filter desktop tauri:dev
```

Python bridgeを直接確認する例:

```powershell
python python/lora_training_gui/bridge.py --job health_check
python python/lora_training_gui/bridge.py --job settings_get
python python/lora_training_gui/bridge.py --job tagger_model_status
```

WD14モデルの直接ダウンロード:

```powershell
$env:PYTHONPATH="$PWD\python"
python -m lora_training_gui.wd14_tagger --model-dir "D:\AI\models\Hot\image\Tagger" --download-only
```

## 検証済み

- `pnpm typecheck`
- `pnpm --filter desktop build`
- `cargo check` in `packages/desktop/src-tauri`
- `python -m compileall python`
- bridge API: settings、job CRUD、TOML生成、training/sample/TensorBoard launch plan
- Dataset API: scan、caption save、bulk tag operation、tag search
- WD14 Tagger: model download、ONNX inference、caption file write

## 関連ドキュメント

- [既存ツール調査](docs/tool-survey.md)
- [プロダクト設計](docs/product-design.md)
- [アーキテクチャ](docs/architecture.md)
- [データと設定](docs/data-and-config.md)
- [実装ステータス](docs/implementation-plan.md)
- [ADR: Tauri + Python Bridge](docs/decisions/0001-tauri-python-bridge.md)
