# データと設定設計

## 設定の階層

設定は4層に分ける。

1. App settings
   - アプリ全体の設定。
   - jobs root、既定エンジン、UI、ログ保持、WanDB既定値。

2. Engine profile
   - 外部学習エンジンの場所と能力。
   - venv、Python、対応モデル系列、対応multi-GPUモード。

3. Job config
   - ユーザーが編集する学習設定。
   - dataset、training、network、optimizer、sample、logging。

4. Run record
   - 実行時に確定したスナップショット。
   - merged config、launch script、env、ログ、終了コード。

## App Settings

保存先候補:

- 開発中: リポジトリ外のローカル設定。
- 製品版: `%LOCALAPPDATA%\LoraTrainingGui\settings.json`

例:

```json
{
  "jobsRoot": "D:\\LoraTraining\\jobs",
  "defaultEngineId": "anima-standalone-local",
  "ui": {
    "theme": "system",
    "logTailLines": 2000
  },
  "wandb": {
    "mode": "disabled",
    "project": "",
    "entity": "",
    "runNameTemplate": "{job}-{date}-{time}"
  }
}
```

## Engine Profile

例:

```json
{
  "id": "anima-standalone-local",
  "type": "anima_standalone",
  "name": "Anima Standalone Trainer",
  "root": "D:\\tool\\lora_trainer\\Anima-Standalone-Trainer",
  "venv": "D:\\tool\\lora_trainer\\Anima-Standalone-Trainer\\venv",
  "python": "D:\\tool\\lora_trainer\\Anima-Standalone-Trainer\\venv\\Scripts\\python.exe",
  "capabilities": {
    "models": ["anima", "lumina"],
    "multiGpuModes": ["single", "ddp", "fsdp", "fsdp2", "deepspeed", "tp_sp"],
    "sampleGeneration": true,
    "persistentGeneration": true,
    "tensorboard": true,
    "wandb": true
  }
}
```

## Job Config

ジョブ編集用の主設定はJSONで管理し、エンジン実行用にTOMLへ変換する。sd-scripts/Anima側の互換性が重要なため、実行直前には必ずTOMLスナップショットを保存する。

例:

```json
{
  "schemaVersion": 1,
  "name": "my_anima_lora",
  "engineId": "anima-standalone-local",
  "architecture": "anima",
  "modelPaths": {
    "ditPath": "D:\\models\\anima-preview.safetensors",
    "qwen3Path": "D:\\models\\qwen_3_06b_base.safetensors",
    "vaePath": "D:\\models\\qwen_image_vae.safetensors"
  },
  "dataset": {
    "configPath": "dataset.toml",
    "captionExtension": ".txt",
    "enableBucket": true,
    "resolution": [1536, 1536]
  },
  "training": {
    "outputName": "my_anima_lora",
    "maxTrainEpochs": 15,
    "learningRate": 0.0001,
    "textEncoderLr": 0.00005,
    "mixedPrecision": "bf16",
    "gradientCheckpointing": true,
    "seed": 42
  },
  "optimizer": {
    "type": "AdamW8bit",
    "args": ["weight_decay=0.01"]
  },
  "network": {
    "module": "networks.lora_anima",
    "dim": 16,
    "alpha": 16,
    "trainUnetOnly": true
  },
  "gpu": {
    "ids": ["0"],
    "mode": "single"
  },
  "sample": {
    "enabled": true,
    "everyNEpochs": 1,
    "promptsPath": "sample_prompts.txt"
  },
  "logging": {
    "tensorboard": true,
    "wandbMode": "disabled"
  }
}
```

## Dataset TOML

実行エンジンにはTOMLを渡す。

初期形:

```toml
[general]
enable_bucket = true
bucket_no_upscale = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

[[datasets]]
resolution = [1536, 1536]
batch_size = 1
caption_extension = ".txt"

  [[datasets.subsets]]
  image_dir = ""
  num_repeats = 1
  keep_tokens = 1
  flip_aug = false
  caption_prefix = ""
  shuffle_caption = false
  caption_tag_dropout_rate = 0.0
  caption_dropout_rate = 0.05
```

## Parameter Registry

Kohya GUIの巨大な `TrainParams` をそのままフォームにしない。UIで扱うため、パラメータ定義をレジストリ化する。

項目ごとに持つ情報:

- key
- label
- section
- type
- default
- min/max/step
- choices
- engine support
- cli flagまたはTOML key
- visible condition
- validation
- help text
- risk level

例:

```json
{
  "key": "optimizer.type",
  "label": "Optimizer",
  "section": "optimizer",
  "type": "select",
  "default": "AdamW8bit",
  "choices": ["AdamW8bit", "AdamW", "Lion", "Lion8bit", "prodigy", "RAdamScheduleFree", "Came", "Custom"],
  "engines": ["sd_scripts", "anima_standalone"],
  "toml": "training_arguments.optimizer_type"
}
```

## Run Record

run recordは再現性のために必ず保存する。

```json
{
  "runId": "20260621-160000",
  "jobName": "my_anima_lora",
  "startedAt": "2026-06-21T16:00:00+09:00",
  "finishedAt": null,
  "status": "running",
  "engineId": "anima-standalone-local",
  "gpuIds": ["0", "1"],
  "multiGpuMode": "fsdp2",
  "launch": {
    "cwd": "D:\\tool\\lora_trainer\\Anima-Standalone-Trainer",
    "scriptPath": "launch.ps1",
    "mergedConfigPath": "merged_config.toml"
  },
  "artifacts": {
    "stdout": "stdout.log",
    "stderr": "stderr.log",
    "metrics": "metrics.jsonl"
  }
}
```

## 秘密情報

Gitに入れないもの:

- WanDB API key。
- Hugging Face token。
- Civitai token。
- ローカルモデルパス入りの個人設定。
- 学習済みweight。
- dataset画像。

`.gitignore` では、jobs、output、logs、venv、weight拡張子を除外する。
