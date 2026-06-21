# LoraTraining

Tauri + TypeScript + Python bridge 縺ｧ菴懊ｋ LoRA 蟄ｦ鄙竪UI縺ｧ縺吶ＡD:\tool\lora_trainer\` 驟堺ｸ九・譌｢蟄倥ヤ繝ｼ繝ｫ繧貞､夜Κ繧ｨ繝ｳ繧ｸ繝ｳ縺ｨ縺励※謇ｱ縺・√ョ繝ｼ繧ｿ繧ｻ繝・ヨ謨ｴ蛯吶√ち繧ｰ邱ｨ髮・√ず繝ｧ繝冶ｨｭ螳壹ゝOML逕滓・縲∝ｭｦ鄙偵・繧ｵ繝ｳ繝励Ν繝ｻTensorBoard襍ｷ蜍輔ｒ1縺､縺ｮ繝・せ繧ｯ繝医ャ繝励い繝励Μ縺ｫ縺ｾ縺ｨ繧√∪縺吶・
## 螳溯｣・ｸ医∩

- Dashboard: Python/Node/pnpm/Cargo/Git/GPU/譌｢蟄倥お繝ｳ繧ｸ繝ｳ讀懷・
- Dataset Studio: 逕ｻ蜒上→caption縺ｮ繧ｹ繧ｭ繝｣繝ｳ縲∝句挨caption菫晏ｭ倥∵ｬ謳・遨ｺ/蟄､遶議aption讀懷・縲∽ｽ手ｧ｣蜒丞ｺｦ讀懷・
- Bulk Tag Edit: 繧ｿ繧ｰ霑ｽ蜉縲∝炎髯､縲∫ｽｮ謠帙∝・鬆ｭ遘ｻ蜍輔《huffle
- Jobs: 繧ｸ繝ｧ繝紋ｽ懈・縲∫ｷｨ髮・∽ｿ晏ｭ倥∬､・｣ｽ縲∝炎髯､縲～dataset.toml` 縺ｨ `_merged_config.toml` 逕滓・
- Training Launch: Anima Standalone蜷代￠ `accelerate launch` 繝励Λ繝ｳ逕滓・縲√・繝ｫ繝；PU繝｢繝ｼ繝峨仝anDB迺ｰ蠅・､画焚縲√・繝ｭ繧ｻ繧ｹ髢句ｧ・蛛懈ｭ｢/繝ｭ繧ｰ陦ｨ遉ｺ
- Samples: `anima_gen.py` 逕ｨ繧ｵ繝ｳ繝励Ν逕滓・繝励Λ繝ｳ縲√・繝ｭ繧ｻ繧ｹ髢句ｧ・蛛懈ｭ｢/繝ｭ繧ｰ陦ｨ遉ｺ
- TensorBoard: 遨ｺ縺阪・繝ｼ繝域､懷・縲∬ｵｷ蜍輔ゞRL陦ｨ遉ｺ
- WD14 Tagger: 繝｢繝・Ν迥ｶ諷狗｢ｺ隱阪∽ｾ晏ｭ伜ｰ主・繝励Λ繝ｳ縲√Δ繝・ΝDL縲√ョ繝ｼ繧ｿ繧ｻ繝・ヨ縺ｸ縺ｮ閾ｪ蜍輔ち繧ｰ莉倥￠

## 繝｢繝・Ν驟咲ｽｮ

WD14 Tagger縺ｮ譌｢螳夐・鄂ｮ蜈・

```text
D:\AI\models\Hot\image\Tagger\
```

縺薙・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｫ莉･荳九ｒ驟咲ｽｮ縺励∪縺吶・
- `model.onnx`
- `selected_tags.csv`

GUI縺ｮ Tagger 逕ｻ髱｢縺九ｉ繧ゅム繧ｦ繝ｳ繝ｭ繝ｼ繝峨〒縺阪∪縺吶・
## 髢狗匱繧ｳ繝槭Φ繝・
```powershell
pnpm install
pnpm typecheck
pnpm --filter desktop build
pnpm --filter desktop tauri:dev
```

Python bridge繧堤峩謗･遒ｺ隱阪☆繧倶ｾ・

```powershell
python python/lora_training_gui/bridge.py --job health_check
python python/lora_training_gui/bridge.py --job settings_get
python python/lora_training_gui/bridge.py --job tagger_model_status
```

WD14繝｢繝・Ν縺ｮ逶ｴ謗･繝繧ｦ繝ｳ繝ｭ繝ｼ繝・

```powershell
$env:PYTHONPATH="$PWD\python"
python -m lora_training_gui.wd14_tagger --model-dir "D:\AI\models\Hot\image\Tagger" --download-only
```

## 讀懆ｨｼ貂医∩

- `pnpm typecheck`
- `pnpm --filter desktop build`
- `cargo check` in `packages/desktop/src-tauri`
- `python -m compileall python`
- bridge API: settings縲）ob CRUD縲ゝOML逕滓・縲》raining/sample/TensorBoard launch plan
- Dataset API: scan縲…aption save縲｜ulk tag operation縲》ag search
- WD14 Tagger: model download縲＾NNX inference縲…aption file write

## 髢｢騾｣繝峨く繝･繝｡繝ｳ繝・
- [譌｢蟄倥ヤ繝ｼ繝ｫ隱ｿ譟ｻ](docs/tool-survey.md)
- [繝励Ο繝繧ｯ繝郁ｨｭ險・(docs/product-design.md)
- [繧｢繝ｼ繧ｭ繝・け繝√Ε](docs/architecture.md)
- [繝・・繧ｿ縺ｨ險ｭ螳咯(docs/data-and-config.md)
- [螳溯｣・ｨ育判](docs/implementation-plan.md)
- [ADR: Tauri + Python Bridge](docs/decisions/0001-tauri-python-bridge.md)

