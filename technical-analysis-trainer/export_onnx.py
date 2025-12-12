# FIX scaler save to a model and export into onnx
import torch

def export_model(model, features, eval_results, output_dir, version, horizon):
    model.eval()
    # dummy_input = torch.randn(1, 30, len(features))  # 1 последовательность, seq_len=30, features
    onnx_path = f"{output_dir}/model_{version}.onnx"

    dummy_input = torch.randn(1, 30, len(features))  # 30 — фиксированное окно
    torch.onnx.export(
        model,
        dummy_input, # type: ignore
        onnx_path,
        input_names=["X"],
        output_names=["y"],
        opset_version=18,  # рекомендуемый для PyTorch 2.x
        dynamo=False,
    )
    print(f"✅ Model exported to {onnx_path}")