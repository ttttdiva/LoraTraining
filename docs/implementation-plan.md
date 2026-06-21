# 実装ステータス

## 目的

LoraTrainingは、LoRA学習の準備、caption/tag編集、学習設定、実行、監視、サンプル生成、WD14自動タグ付けをTauriデスクトップアプリにまとめる。

## 実装済みフェーズ

### 1. Desktop Shell

- Tauri + Vite + TypeScript のアプリ骨格
- Dashboard、Dataset Studio、Jobs、WD14 Tagger の画面
- Rust command から Python bridge を呼ぶIPC

### 2. Environment / Settings

- `settings_get` / `settings_save`
- 既定 jobs root: `data/jobs`
- 既定Taggerモデル: `D:\AI\models\Hot\image\Tagger`
- `D:\tool\lora_trainer\Anima-Standalone-Trainer` を初期エンジンとして登録
- Python、Node、pnpm、Cargo、Git、nvidia-smi、GPU情報、既存エンジン候補を検出

### 3. Dataset Studio

- dataset scan
- image/caption対応確認
- caption editor
- caption保存
- タグ追加、削除、置換、先頭移動、shuffle
- タグ検索
- 欠損caption、空caption、孤立txt、低解像度検出

### 4. Jobs / Config Generation

- job作成、取得、保存、複製、削除
- model paths、dataset、training、network、GPU、sample、WanDB設定のGUI編集
- `dataset.toml`
- `sample_prompts.txt`
- `_merged_config.toml`
- output/logs/samplesディレクトリ作成

### 5. Launch / Process Management

- training launch plan
- sample launch plan
- TensorBoard launch plan
- Rust側process registry
- stdout/stderrログ保持
- start/stop/status
- Windowsではprocess tree停止に `taskkill /T /F` を使用

### 6. Multi GPU / WanDB

- GPU ID指定
- single、ddp、fsdp、fsdp2、deepspeed のaccelerate flags生成
- Windows複数GPU向け `USE_LIBUV=0`、`MASTER_ADDR`、`MASTER_PORT`
- `CUDA_VISIBLE_DEVICES`
- `WANDB_MODE`、`WANDB_PROJECT`、`WANDB_ENTITY`

### 7. WD14 Tagger

- `wd-v1-4-convnext-tagger-v2.onnx` / `wd-v1-4-convnext-tagger-v2-selected_tags.csv` の状態確認
- 依存導入プラン
- HuggingFaceからのモデルダウンロード
- ONNX Runtimeでのタグ推論
- merge / overwriteでcaption書き込み

## 検証

- `pnpm typecheck`
- `pnpm --filter desktop build`
- `cargo check` in `packages/desktop/src-tauri`
- `python -m compileall python`
- settings bridge
- job CRUD
- TOML生成
- training/sample/TensorBoard launch plan
- dataset scan
- caption save
- bulk tag operation
- tag search
- WD14 model download
- WD14 ONNX inference
- WD14 caption file write

## 外部依存

実学習とサンプル生成は、外部エンジン、ベースモデル、VAE、text encoder、dataset、GPU/CUDA/Torch環境に依存する。GUI側では起動コマンド生成と事前エラー表示までを担当し、実行ログはprocess registry経由で表示する。
