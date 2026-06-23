# LoraTraining

Tauri + TypeScript + Python bridge で作る LoRA 学習GUIです。必要な学習エンジンコードはリポジトリ内の `engines/sd-scripts` に同梱し、データセット整備、タグ編集、ジョブ設定、TOML生成、学習、サンプル生成、TensorBoard起動を1つのデスクトップアプリにまとめます。

## 実装済み

- Dashboard: Python / Node.js / pnpm / Cargo / Git / GPU / 内部エンジン検証
- Dataset Studio: 画像とcaptionのスキャン、個別caption保存、欠損/空/孤立caption検出、画像解像度検出
- Bulk Tag Edit: タグ追加、削除、置換、先頭移動、shuffle
- Jobs: ジョブ作成、編集、保存、複製、削除、`dataset.toml` と `_merged_config.toml` 生成
- Training Launch: 同梱 sd-scripts の `accelerate launch` プラン生成、マルチGPUモード、WanDB環境変数、プロセス開始/停止、ログ表示
- Samples: 同梱 engine の `anima_gen.py` / `gen_img.py` / `sdxl_gen_img.py` 向けサンプル生成プラン、プロセス開始/停止、ログ表示
- TensorBoard: 空きポート検出、起動、URL表示
- WD14 Tagger: モデル状態確認、依存導入プラン、モデルDL、データセットへの自動タグ付け

## エンジン配置

既定ではリポジトリ内の同梱 engine を使います。

```text
engines/sd-scripts
```

- Anima学習: `anima_train_network.py` / `anima_gen.py`
- SD1/SD2学習: `train_network.py`
- SDXL学習: `sdxl_train_network.py`
- サンプル生成: `gen_img.py` / `sdxl_gen_img.py`
- ComfyUI変換: `networks/convert_anima_lora_to_comfy.py`

Python環境は `engines/sd-scripts/venv` に作成します。venv、学習成果物、データセット、ログ、モデル重みは Git 管理外です。

## WD14 Taggerモデル配置

既定の配置先:

```text
data/models/wd14
```

必要ファイル:

- `wd-v1-4-convnext-tagger-v2.onnx`
- `wd-v1-4-convnext-tagger-v2-selected_tags.csv`

これは `SmilingWolf/wd-v1-4-convnext-tagger-v2` 由来の `WD14 ConvNeXt Tagger v2` です。GUIの Tagger 画面からもダウンロードできます。旧名 `model.onnx` / `selected_tags.csv` が残っている場合は、実行時に明示名へ自動移行します。

## 開発コマンド

```powershell
pnpm install
pnpm typecheck
pnpm --filter desktop build
pnpm --filter desktop tauri:dev
```

Python bridgeを直接確認する例:

```powershell
$env:PYTHONPATH="$PWD\python"
python python/lora_training_gui/bridge.py --job health_check
python python/lora_training_gui/bridge.py --job settings_get
python python/lora_training_gui/bridge.py --job tagger_model_status
```

WD14モデルの直接ダウンロード:

```powershell
$env:PYTHONPATH="$PWD\python"
python -m lora_training_gui.wd14_tagger --model-dir "data/models/wd14" --download-only
```

## 検証済み

- `pnpm typecheck`
- `pnpm --filter desktop build`
- `cargo check` in `packages/desktop/src-tauri`
- `python -m compileall python`
- bridge API: settings、job CRUD、TOML生成、training/sample/TensorBoard launch plan
- sd-scripts: `train_network.py` / `sdxl_train_network.py` / `gen_img.py` / `sdxl_gen_img.py` の `--help` 起動確認
- sd-scripts: 生成した `dataset.toml` を `library.config_util` でschema検証
- Dataset API: scan、caption save、bulk tag operation、tag search
- WD14 Tagger: model download、ONNX inference、caption file write

## 関連ドキュメント

- [内部エンジン構成](docs/tool-survey.md)
- [プロダクト設計](docs/product-design.md)
- [アーキテクチャ](docs/architecture.md)
- [データと設定](docs/data-and-config.md)
- [実装計画](docs/implementation-plan.md)
- [ADR: Tauri + Python Bridge](docs/decisions/0001-tauri-python-bridge.md)
