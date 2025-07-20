# Splendor/Play/export_trained_model.py
"""
Takes a .keras model you've trained and assigns
it as the .onnx model used for any Play here.
"""

import sys
import shutil
import json
import importlib.util
from pathlib import Path


def convert_to_onnx(keras_model_path: Path, onnx_output_path: Path):
    """Convert keras model to ONNX."""
    spec = importlib.util.find_spec("tf2onnx")
    if spec is None:
        return False  # tf2onnx not installed

    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(keras_model_path)
    proto, _ = tf2onnx.convert.from_keras(
        model, output_path=str(onnx_output_path)
    )
    print(f"Exported ONNX to {onnx_output_path.relative_to(Path.cwd())}")

    return True


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m Splendor.Play.export_trained_model "
                 "<path/to/model.keras>")

    keras_model_path = Path(sys.argv[1]).expanduser()
    assert keras_model_path.exists(), f"{keras_model_path} not found"

    # Runtime directory
    runtime_directory = Path(__file__).parent / "runtime"
    runtime_directory.mkdir(exist_ok=True)

    # Convert model
    onnx_output_path = runtime_directory / "model.onnx"
    if convert_to_onnx(keras_model_path, onnx_output_path):
        model_metadata = {
            "format": "onnx",
            "filename": onnx_output_path.name
        }
    else:
        keras_dest_path = runtime_directory / keras_model_path.name
        shutil.copy(keras_model_path, keras_dest_path)
        print(f"Copied Keras model to {keras_dest_path.relative_to(Path.cwd())}")
        model_metadata = {
            "format": "keras",
            "filename": keras_dest_path.name
        }

    # Update GUI load file
    model_json_path = runtime_directory / "current_model.json"
    model_json_path.write_text(json.dumps(model_metadata))
    print("Updated runtime/current_model.json")


if __name__ == "__main__":
    main()
