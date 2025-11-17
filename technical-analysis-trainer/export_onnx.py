# py-trainer/export_onnx.py
import json, os
from onnxmltools import convert_lightgbm # type: ignore
from skl2onnx.common.data_types import FloatTensorType # type: ignore

def export_model(model, features, eval_results, output_dir, version): # type: ignore
    # модель — lgb.LGBMRegressor, у неё есть booster_
    booster = model.booster_ # type: ignore

    # конвертация LightGBM → ONNX
    onnx_model = convert_lightgbm( # type: ignore
        booster, # type: ignore
        initial_types=[("float_input", FloatTensorType([None, len(features)]))], # type: ignore
        target_opset=15
    )

    model_path = os.path.join(output_dir, version) # type: ignore
    os.makedirs(model_path, exist_ok=True) # type: ignore
    onnx_file = os.path.join(model_path, "model.onnx") # type: ignore

    with open(onnx_file, "wb") as f:
        f.write(onnx_model.SerializeToString()) # type: ignore

    meta = { # type: ignore
        "model_name": "price_regressor_lgbm",
        "version": version,
        "features": features,
        "metrics": eval_results,
    }
    with open(os.path.join(model_path, "meta.json"), "w") as f: # type: ignore
        json.dump(meta, f, indent=2)

    print(f"✅ ONNX model exported to {onnx_file}")