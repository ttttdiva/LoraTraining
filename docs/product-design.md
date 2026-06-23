# プロダクト設計

## 目的

LoRA学習に必要な準備、設定、実行、監視、サンプル確認を1つのデスクトップGUIにまとめる。

対象ユーザーは、sd-scriptsやAnima系トレーナーを使いたいが、毎回コマンド、TOML、venv、GPU割り当て、タグ編集を手作業で管理したくないユーザー。

## 成功条件

- 初回起動から環境診断、モデルパス設定、データセット準備、学習開始までGUIで完結する。
- sd-scripts相当の主要パラメータを隠さず、初心者向けプリセットと詳細編集を切り替えられる。
- AdamW8bit固定にせず、optimizerとoptimizer_argsを明示的に扱える。
- 1GPU、複数GPU、FSDP、DeepSpeed、TP/SPなどをジョブ単位で選べる。
- 学習中ログ、GPU状態、loss、サンプル、TensorBoard、WanDB状態を同じジョブ画面で追える。
- 実行した設定、生成コマンド、ログ、成果物が後から再現できる。

## 非ゴール

- 既存3ツールを1つの画面に埋め込むだけのラッパーにはしない。
- 学習エンジン自体の全面再実装はしない。
- 最初から全モデル系列を完全対応しようとしない。
- クラウド学習サービスにはしない。

## 主要画面

### 1. Dashboard

- 最近のジョブ一覧。
- 実行中ジョブの状態。
- GPU状態。
- エンジン環境の健康状態。

### 2. Project / Job

- ジョブ名、出力名、モデル種別、エンジン選択。
- dataset、model paths、training、network、optimizer、sample、loggingをタブで編集。
- 「簡易」「詳細」「raw TOML」の3段階表示。
- 起動前検証とコマンドプレビュー。

### 3. Dataset Studio

- 画像とcaption txtの対応確認。
- サムネイル付きcaption/tag編集。
- WD14などの自動タグ付けジョブ。
- タグ置換、削除、先頭移動、shuffle、keep token確認。
- 低解像度、caption欠落、孤立txt、重複画像の検出。
- bucket分布プレビュー。

### 4. Training Monitor

- stdout/stderrログ。
- loss、step、epoch、残り時間。
- GPU/CPU/RAM/VRAM/温度。
- pause/stopはエンジンが安全に対応できる範囲で提供。
- resume state検出。

### 5. Samples

- sample_prompts編集。
- 学習中に生成されたサンプル一覧。
- 任意checkpoint/LoRA weightでone-shot生成。
- keep-loaded生成サーバーは必要時のみ動かし、ポートは動的割り当て。

### 6. Environment

- Python、Git、CUDA、NVIDIA driver、Torch、venv、内部 engine rootを検証。
- 同梱 `engines/sd-scripts` を built-in engine として登録。
- エンジンごとのセットアップ、更新、検証を行う。

### 7. Settings

- グローバルモデルパス。
- 既定のjobs root。
- WanDB設定。
- TensorBoard設定。
- UIテーマとログ保持日数。

## ユーザーフロー

### 初回セットアップ

1. アプリ起動。
2. Environment画面で既存エンジンを検出。
3. Python/venv/Torch/CUDA/GPUを診断。
4. モデルパスを登録。
5. デフォルトjobs rootを選択。

### 新規学習

1. New Jobを作成。
2. エンジンとモデル系列を選ぶ。
3. Dataset Studioで画像とcaptionを整える。
4. プリセットから学習設定を選び、必要なら詳細を調整。
5. sample promptsを設定。
6. Preflightで不足や危険設定を確認。
7. Start Training。

### 再開と比較

1. 過去ジョブを開く。
2. 保存stateやcheckpointを選ぶ。
3. 設定差分を確認。
4. resumeまたはcloneして別条件で再実行。

## 設定の見せ方

基本は「安全な少数項目」から始める。

- Basic: dataset、output、epoch/step、learning rate、network dim/alpha、optimizer、precision、GPU。
- Advanced: scheduler、dropout、bucket、cache、block swap、attention、resume、sample cadence。
- Expert: engine固有TOML、追加引数、custom optimizer、FSDP/DeepSpeed詳細。

## UXで避けること

- 学習開始前に黒いコンソールだけを見せてユーザーに判断させること。
- 詳細設定を完全に隠して、問題発生時に逃げ場をなくすこと。
- 1つの巨大フォームに全パラメータを並べること。
- 生成されたコマンドだけが唯一の保存状態になること。
