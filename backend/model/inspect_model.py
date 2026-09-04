"""
inspect_model.py

The trained model files in this folder (steel_weights.weights.h5,
steel_defect_mobilenet.h5, steel_defect_mobilenet.keras) are BINARY files.
They store millions of trained numeric weights in the HDF5 / Keras zip
format. They are not source code and will show up as garbled control
characters if opened directly in a text editor like VS Code -- that is
expected and does not mean the file is broken.

Run this script to generate a clean, human-readable TEXT summary of what
is actually stored inside the binary weight file:

    cd backend/model
    python inspect_model.py

It writes MODEL_SUMMARY.txt in this same folder.
"""

import h5py

SOURCE_FILE = "steel_weights.weights.h5"
OUTPUT_FILE = "MODEL_SUMMARY.txt"


def main():
    lines = []
    total_params = 0

    with h5py.File(SOURCE_FILE, "r") as f:
        def visit(name, obj):
            nonlocal total_params
            if isinstance(obj, h5py.Dataset):
                param_count = 1
                for dim in obj.shape:
                    param_count *= dim
                total_params += param_count
                lines.append(
                    f"{name:70s} shape={str(obj.shape):20s} "
                    f"dtype={str(obj.dtype):10s} params={param_count:,}"
                )

        f.visititems(visit)

    with open(OUTPUT_FILE, "w") as out:
        out.write("Steel Defect Analysis Model -- Weight Summary\n")
        out.write("=" * 60 + "\n")
        out.write(f"Source file: {SOURCE_FILE}\n")
        out.write(f"Total weight tensors: {len(lines)}\n")
        out.write(f"Total trainable parameters: {total_params:,}\n")
        out.write("=" * 60 + "\n\n")
        out.write("\n".join(lines))
        out.write("\n")

    print(f"Wrote {OUTPUT_FILE} ({len(lines)} tensors, {total_params:,} params)")


if __name__ == "__main__":
    main()
