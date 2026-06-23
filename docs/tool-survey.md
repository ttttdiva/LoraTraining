# Internal Engine Layout

The application ships the runtime engine code in this repository.

```text
engines/
  sd-scripts/
    train_network.py
    sdxl_train_network.py
    anima_train_network.py
    anima_gen.py
    gen_img.py
    sdxl_gen_img.py
    library/
    networks/
    tools/
    finetune/
    configs/
    pytorch_lightning/
```

The bundled tree intentionally excludes git metadata, venvs, caches, tests, docs, images, datasets, logs, and model artifacts.

## Runtime Profiles

- `anima-standalone`: uses the bundled engine root for Anima LoRA training and sample generation.
- `sd-scripts`: uses the same bundled engine root for SD1/SD2/SDXL LoRA training and sample generation.

Both profiles create and use `engines/sd-scripts/venv`.

## Required Runtime Files

- Training: `anima_train_network.py`, `train_network.py`, `sdxl_train_network.py`
- Samples: `anima_gen.py`, `gen_img.py`, `sdxl_gen_img.py`
- Conversion: `networks/convert_anima_lora_to_comfy.py`
- Imports: `library/`, `networks/`, `tools/`, `finetune/`, `configs/`, `pytorch_lightning/`
- Setup: `requirements.txt`, `setup.py`, `LICENSE.md`
