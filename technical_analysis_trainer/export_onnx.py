import torch
from typing import Any


def export_model(
    model, features, eval_results, output_dir, version, horizon, threshold: float
):
    """
    horizon здесь — длина окна (seq_len), как в config window.long.
    """
    model.eval()
    onnx_path = f"{output_dir}/model_{version}.onnx"

    dummy_input = torch.randn(1, horizon, len(features))  # (batch, seq_len, features)
    print("dummy_input", dummy_input.shape)
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        onnx_path,
        input_names=["X"],
        output_names=["y"],
        opset_version=18,  # рекомендуемый для PyTorch 2.x
        dynamo=False,
    )
    print(f"✅ Model exported to {onnx_path}")
