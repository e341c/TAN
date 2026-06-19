import torch


def export_model(model, features, eval_results, output_dir, version, horizon):
    """
    horizon — sequence length (seq_len), as in config window.long.
    """
    model.eval()
    onnx_path = f"{output_dir}/model_{version}.onnx"

    dummy_input = torch.randn(1, horizon, len(features))
    print("dummy_input", dummy_input.shape)
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        onnx_path,
        input_names=["X"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
    )
    print(f"✅ Model exported to {onnx_path}")
