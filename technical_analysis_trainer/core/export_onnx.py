import json
import os

import torch


def export_model(
    model,
    features,
    eval_results,
    output_dir,
    version,
    horizon,
    save_state_dict: bool = True,
    batch_size: int = 8,
    provenance: dict | None = None,
):
    """
    Export to ONNX with a dynamic batch axis.

    horizon — sequence length (seq_len), as in config window.long.

    The batch dimension is declared dynamic so the graph accepts any number of
    windows. Exporting with a fixed batch of 1 (the previous behaviour) makes
    bulk scoring impossible: every backtest, sweep, or evaluation has to loop one
    window at a time, and batched calls fail outright with
    "Got invalid dimensions for input: X".

    The dummy input deliberately uses batch > 1 — tracing with batch 1 lets
    size-1 dimensions get baked into the graph even when a dynamic axis is
    declared.

    save_state_dict also writes the torch weights, so the model can be
    re-exported later (different opset, dynamic axes, quantisation) without
    retraining. Artifacts previously kept only the ONNX, which made any export
    change a full retrain.
    """
    model.eval()
    onnx_path = f"{output_dir}/model_{version}.onnx"

    dummy_input = torch.randn(batch_size, horizon, len(features))
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        onnx_path,
        input_names=["X"],
        output_names=["y"],
        dynamic_axes={"X": {0: "batch"}, "y": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"✅ Model exported to {onnx_path} (dynamic batch, seq_len={horizon})")

    if save_state_dict:
        weights_path = f"{output_dir}/weights_{version}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "features": list(features),
                "seq_len": horizon,
                "model_class": type(model).__name__,
            },
            weights_path,
        )
        print(f"✅ Weights saved to {weights_path}")

    meta_path = f"{output_dir}/meta_{version}.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": version,
                    "seq_len": horizon,
                    "n_features": len(features),
                    "features": list(features),
                    "dynamic_batch": True,
                    "opset": 18,
                    # What this artifact was allowed to see. Without it the only
                    # durable record of a model's training span is its export
                    # stamp, which bounds the answer but does not give it — and
                    # every "out-of-sample" claim depends on knowing it exactly.
                    "provenance": _jsonable(provenance or {}),
                    "eval": _jsonable(eval_results),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as e:
        print(f"⚠ could not write {meta_path}: {e}")


def _jsonable(obj):
    """Best-effort conversion of eval dicts (numpy scalars, tuples, models)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not _is_model(v)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "size", 1) == 1:
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _is_model(v) -> bool:
    return isinstance(v, torch.nn.Module)


def verify_dynamic_batch(onnx_path: str, seq_len: int, n_features: int) -> bool:
    """Confirm the exported graph really accepts more than one window."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    for batch in (1, 4, 32):
        x = np.random.randn(batch, seq_len, n_features).astype(np.float32)
        out = sess.run(None, {name: x})[0]
        if out.shape[0] != batch:
            raise AssertionError(f"batch {batch} -> output {out.shape}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("usage: export_onnx.py <onnx_path> <seq_len> <n_features>")
        raise SystemExit(2)
    path, seq_len, n_feat = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    verify_dynamic_batch(path, seq_len, n_feat)
    print(f"✅ {os.path.basename(path)} accepts dynamic batch")
