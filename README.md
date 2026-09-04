# Defect Analysis System — TATA Steel

An AI-powered industrial surface inspection tool. A user uploads an image of
a steel surface, and the system identifies the defect category and displays
it on screen with a confidence score.

Built to the brief in *Project 1: Defect Analysis System* (Corporate
Functions):

- User uploads an image of a defect on a steel surface
- The model classifies the image against a predefined defect catalogue
- The result — the defect name — is displayed on screen
- Frontend: React JS upload screen
- Backend: calls the AI model and returns the result to the frontend

---

## Overview

The system uses a MobileNetV2-based convolutional neural network (transfer
learning), fine-tuned to classify steel surface images into one of six
defect categories. The frontend is a React application; the backend is a
FastAPI service that loads the trained model and serves predictions over a
REST endpoint.

## Defect catalogue

The model classifies images into these categories:

| Defect | Description |
|---|---|
| Crazing | Fine network of surface cracks from thermal/mechanical stress |
| Inclusion | Foreign material (e.g. slag, oxide) trapped in the surface |
| Patches | Localized irregular regions with a different surface finish |
| Pitted Surface | Small cavities or pits, typically from corrosion |
| Rolled-in Scale | Oxide scale pressed into the surface during hot rolling |
| Scratches | Linear marks from friction or mechanical abrasion |

> Note: these are the six defect classes the model was actually trained on
> (the standard steel-surface defect set). If the catalogue needs to be
> expanded to include additional categories, the model must be retrained
> with labeled examples for that category — this is a data/training task,
> not a frontend change.

## Tech stack

**Frontend:** React, Vite, Axios
**Backend:** FastAPI, TensorFlow / Keras, NumPy, Pillow
**Model:** MobileNetV2 (transfer learning) trained on labeled steel surface
images

## Project structure

```text
.
├── frontend/           React application (upload UI + result display)
│   ├── src/
│   ├── public/
│   └── package.json
├── backend/            FastAPI service
│   ├── main.py
│   ├── model/          trained weights (.h5 / .keras)
│   ├── requirements.txt
│   └── trainning.ipynb / trainning_new.ipynb   training notebooks
└── runtime.txt
```

## Local setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Runs at `http://127.0.0.1:8000`. Interactive API docs at
`http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173`. Set `VITE_API_URL` in `frontend/.env` to
point at the backend (defaults to `http://127.0.0.1:8000` for local
development).

## API

**POST `/predict`** — multipart form upload, field name `file`

Response:

```json
{
  "prediction": "scratches",
  "confidence": 98.45
}
```

**GET `/health`** — service health check

## A note on the model files

`backend/model/*.h5` and `*.keras` are **binary files** containing ~2.6
million trained numeric weights — not source code. If one is opened
directly in a text editor it will show scrambled characters; that's normal
for any binary file and doesn't mean it's broken. See
`backend/model/README.md` for how to inspect them properly as readable
text (`python inspect_model.py` generates `MODEL_SUMMARY.txt`).

## Team

Project guide: Mr. Vivudh Fore

Team: Datla Sandeep Varma, Varri Shyam Sai, Varshit Atuluri,
G.D.V.S. Sai Ganesh

B.Tech CSE, Gurukula Kangri Vishwavidyalaya, Haridwar

## Future improvements

- Additional defect categories (requires labeled training data + retraining)
- Model explainability (e.g. Grad-CAM overlays on the uploaded image)
- Inspection history / logging per session
- Batch upload for multiple images at once
