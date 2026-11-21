
import os
import urllib.request
import json
import numpy as np
#pip install essentia-tensorflow
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D, TensorflowPredictMusiCNN, TempoCNN


BASE = "https://essentia.upf.edu/models"
MODELS = {
    "embedding":      "feature-extractors/discogs-effnet/discogs-effnet-bs64-1",
    "genre":          "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1",
    "happy":          "classification-heads/mood_happy/mood_happy-discogs-effnet-1",
    "sad":            "classification-heads/mood_sad/mood_sad-discogs-effnet-1",
    "relaxed":        "classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1",
    "aggressive":     "classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1",
    "party":          "classification-heads/mood_party/mood_party-discogs-effnet-1",
    "danceability":   "classification-heads/danceability/danceability-discogs-effnet-1",
    "voice":          "classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1",
    "instruments":    "classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1",
    "timbre":         "classification-heads/timbre/timbre-discogs-effnet-1",                     
    "acoustic":       "classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1",
    "electronic":     "classification-heads/mood_electronic/mood_electronic-discogs-effnet-1",
    "deam":           "classification-heads/deam/deam-msd-musicnn-2",
    "musicnn":        "feature-extractors/musicnn/msd-musicnn-1",
    "tempocnn":       "tempo/tempocnn/deeptemp-k16-3",      

}

os.makedirs("models", exist_ok=True)
for name, path in MODELS.items():
    for ext in (".pb", ".json"):
       url = f"{BASE}/{path}{ext}"
       dest = os.path.join("models", os.path.basename(url))
       if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)

def model_files(name):
    fname = os.path.basename(MODELS[name])
    return f"models/{fname}.pb", f"models/{fname}.json"


def load_meta(name):
    x, meta_path = model_files(name)
    with open(meta_path) as f:
        return json.load(f)


def io_names(meta, purpose):
    schema = meta.get("schema", {})
    inputs = schema.get("inputs", [])
    outputs = schema.get("outputs", [])
    inp = inputs[0]["name"]
    out = next((o["name"] for o in outputs if o.get("output_purpose") == purpose))
    return inp, out


_emb_meta = load_meta("embedding")
x, _emb_out = io_names(_emb_meta, "embeddings")
embedder = TensorflowPredictEffnetDiscogs(
    graphFilename=model_files("embedding")[0],
    output=_emb_out or "PartitionedCall:1",
)


def get_embeddings(path):
    audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
    return embedder(audio)


def predict(name, embeddings):
    meta = load_meta(name)
    inp, out = io_names(meta, "predictions")
    kwargs = {"graphFilename": model_files(name)[0]}
    if inp:
        kwargs["input"] = inp
    if out:
        kwargs["output"] = out
    model = TensorflowPredict2D(**kwargs)
    scores = np.mean(model(embeddings), axis=0)
    return dict(zip(meta["classes"], scores.tolist()))


tempo_model = TempoCNN(graphFilename=model_files("tempocnn")[0])

def get_tempo(path):
    audio = MonoLoader(filename=path, sampleRate=11025, resampleQuality=4)()
    global_tempo, local_tempo, _ = tempo_model(audio)
    bpm = float(global_tempo)
    local = [round(float(t)) for t in local_tempo]   
    return {
        "local_bpm": local,
        "global_bpm": round(bpm, 2),
    }

musicnn_embedder = TensorflowPredictMusiCNN(graphFilename=model_files("musicnn")[0],
                                            output="model/dense/BiasAdd")
av_model = TensorflowPredict2D(graphFilename=model_files("deam")[0], output="model/Identity")



def get_arousal_valence(path):
    audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
    emb = musicnn_embedder(audio)
    valence, arousal = np.mean(av_model(emb), axis=0)  
    return {
        "arousal": round(float(arousal), 2),     
        "valence": round(float(valence), 2),    
        "energy": float(arousal),
        "positivity": float(valence),
    }

def analyze(path):
    emb = get_embeddings(path)
    voice = predict("voice", emb)
    return {
        #"genre": predict("genre", emb),
        "mood": {m: predict(m, emb)
                 for m in ("happy", "sad", "relaxed", "aggressive", "party")},
        "danceability": predict("danceability", emb),
        "voice": max(voice, key=voice.get),
        "instruments": predict("instruments", emb),
        "timbre": {c: round(s, 3) for c, s in predict("timbre", emb).items()},
        "tempo": get_tempo(path),                             
        "acoustic": predict("acoustic", emb),     
        "electronic": predict("electronic", emb), 
        "emotion": get_arousal_valence(path),          
    }

p = "sample/sample1.mp3"
result = analyze(p)
print(json.dumps(result, indent=2, ensure_ascii=False))