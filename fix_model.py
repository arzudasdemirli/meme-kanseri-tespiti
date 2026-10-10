import zipfile
import json
import shutil
import os

ORIGINAL_MODEL = "final_densenet121_meme_kanseri.keras"
FIXED_MODEL = "fixed_densenet121_meme_kanseri.keras"
TEMP_DIR = "temp_model_fix"

def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)

with zipfile.ZipFile(ORIGINAL_MODEL, "r") as zip_ref:
    zip_ref.extractall(TEMP_DIR)

config_path = os.path.join(TEMP_DIR, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

remove_quantization_config(config)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f)

with zipfile.ZipFile(FIXED_MODEL, "w", zipfile.ZIP_DEFLATED) as zip_ref:
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, TEMP_DIR)
            zip_ref.write(file_path, arcname)

shutil.rmtree(TEMP_DIR)

print("Model düzeltildi:", FIXED_MODEL)