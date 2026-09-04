# Model files

- `steel_weights.weights.h5` — trained weights only (loaded in `main.py`)
- `steel_defect_mobilenet.h5` — full model (architecture + weights), legacy format
- `steel_defect_mobilenet.keras` — full model, current Keras format

**These three files are binary, not text or code.** They store roughly
2.6 million trained numeric weights in the HDF5 / Keras zip format. If you
open one directly in VS Code (or any text editor), you'll see a wall of
scrambled characters and control-code labels (`NUL`, `SOH`, `STX`, etc.) —
that's expected for any binary file opened as text, and does not mean the
file is broken or corrupted.

To see what's actually inside one in plain, readable text, run:

```bash
python inspect_model.py
```

This generates `MODEL_SUMMARY.txt` — a plain-text listing of every layer
name, shape, and parameter count in the model.

To actually *use* the model, you don't open these files directly at all —
`backend/main.py` loads them programmatically via
`model.load_weights("model/steel_weights.weights.h5")`.
