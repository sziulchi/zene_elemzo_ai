
import json
import os
import urllib.request

import numpy as np

BASE = "https://essentia.upf.edu/models"
MODELS = {
    "embedding":    "feature-extractors/discogs-effnet/discogs-effnet-bs64-1",
    "genre":        "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1",
    "happy":        "classification-heads/mood_happy/mood_happy-discogs-effnet-1",
    "sad":          "classification-heads/mood_sad/mood_sad-discogs-effnet-1",
    "relaxed":      "classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1",
    "aggressive":   "classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1",
    "party":        "classification-heads/mood_party/mood_party-discogs-effnet-1",
    "danceability": "classification-heads/danceability/danceability-discogs-effnet-1",
    "voice":        "classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1",
    "instruments":  "classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1",
    "timbre":       "classification-heads/timbre/timbre-discogs-effnet-1",
    "acoustic":     "classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1",
    "electronic":   "classification-heads/mood_electronic/mood_electronic-discogs-effnet-1",
    "deam":         "classification-heads/deam/deam-msd-musicnn-2",
    "musicnn":      "feature-extractors/musicnn/msd-musicnn-1",
    "tempocnn":     "tempo/tempocnn/deeptemp-k16-3",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_cache = {}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path in MODELS.values():
        for ext in (".pb", ".json"):
            url = f"{BASE}/{path}{ext}"
            dest = os.path.join(MODELS_DIR, os.path.basename(url))
            if not os.path.exists(dest):
                print("Letoltes:", url)
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:
                    # nem minden modellhez van .json (pl. tempocnn), ez nem hiba
                    if ext == ".pb":
                        raise
                    print("  (nincs .json, kihagyva)", e)


def model_files(name):
    fname = os.path.basename(MODELS[name])
    return os.path.join(MODELS_DIR, fname + ".pb"), os.path.join(MODELS_DIR, fname + ".json")


def load_meta(name):
    with open(model_files(name)[1], encoding="utf-8") as f:
        return json.load(f)


def io_names(meta, purpose):
    schema = meta.get("schema", {})
    inputs = schema.get("inputs", [])
    outputs = schema.get("outputs", [])
    inp = inputs[0]["name"] if inputs else None
    out = next((o["name"] for o in outputs if o.get("output_purpose") == purpose), None)
    return inp, out


def _es():
    """Essentia es a modellek lusta betoltese (csak az elso hivaskor)."""
    if "es" not in _cache:
        download_models()
        import essentia.standard as es
        _cache["es"] = es

        _, emb_out = io_names(load_meta("embedding"), "embeddings")
        _cache["embedder"] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model_files("embedding")[0],
            output=emb_out or "PartitionedCall:1",
        )
        _cache["tempo"] = es.TempoCNN(graphFilename=model_files("tempocnn")[0])
        _cache["musicnn"] = es.TensorflowPredictMusiCNN(
            graphFilename=model_files("musicnn")[0], output="model/dense/BiasAdd")
        _cache["deam"] = es.TensorflowPredict2D(
            graphFilename=model_files("deam")[0], output="model/Identity")
    return _cache["es"]


def _load_audio(path, sr):
    return _es().MonoLoader(filename=path, sampleRate=sr, resampleQuality=4)()


def get_embeddings(path):
    _es()
    return _cache["embedder"](_load_audio(path, 16000))


def predict(name, embeddings):
    es = _es()
    key = "head_" + name
    meta = load_meta(name)
    if key not in _cache:
        inp, out = io_names(meta, "predictions")
        kwargs = {"graphFilename": model_files(name)[0]}
        if inp:
            kwargs["input"] = inp
        if out:
            kwargs["output"] = out
        _cache[key] = es.TensorflowPredict2D(**kwargs)
    scores = np.mean(_cache[key](embeddings), axis=0)
    return {c: round(float(s), 3) for c, s in zip(meta["classes"], scores)}


def get_tempo(path):
    _es()
    global_tempo, local_tempo, _ = _cache["tempo"](_load_audio(path, 11025))
    return {
        "global_bpm": round(float(global_tempo), 2),
        "local_bpm": [round(float(t)) for t in local_tempo],
    }


def get_arousal_valence(path):
    _es()
    emb = _cache["musicnn"](_load_audio(path, 16000))
    valence, arousal = np.mean(_cache["deam"](emb), axis=0)
    return {
        "arousal": round(float(arousal), 2),
        "valence": round(float(valence), 2),
    }


def analyze(path):
    emb = get_embeddings(path)
    voice = predict("voice", emb)
    instruments = predict("instruments", emb)
    top_instruments = dict(sorted(instruments.items(), key=lambda kv: -kv[1])[:8])
    return {
        # "genre": predict("genre", emb),   # szandekosan kihagyva: az LLM tippelje
        "mood": {m: predict(m, emb) for m in ("happy", "sad", "relaxed", "aggressive", "party")},
        "danceability": predict("danceability", emb),
        "voice": max(voice, key=voice.get),
        "instruments": top_instruments,
        "timbre": predict("timbre", emb),
        "tempo": get_tempo(path),
        "acoustic": predict("acoustic", emb),
        "electronic": predict("electronic", emb),
        "emotion": get_arousal_valence(path),
    }


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "sample/sample1.mp3"
    print(json.dumps(analyze(p), indent=2, ensure_ascii=False))
