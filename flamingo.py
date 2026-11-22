"""Az mp3 alapjan ismerje fel a zenet, vegye ki a nyers tulajdonsagait(stilus, utem, hangulat, hangszin stb.) utana az alapjan (chat elozmeny es mufaj megnevezese nelkul) tulajdonsagabol probalja meg felismerni a mufajt aztan adjon hasonlokat ."""
from pathlib import Path
import json
import ast

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

MINTAFAJL = Path(__file__).resolve().parent / "sample" / "sample1.mp3"


def hangfajl_ellenorzese(utvonal):
    hangfajl = Path(utvonal)
    if not hangfajl.is_file():
        raise FileNotFoundError(f"A hangfájl nem található. {hangfajl}")
    return hangfajl


def kerdezd_flamingot(hangfajl, kerdes):
    kliens = Client("https://nvidia-music-flamingo.hf.space", verbose=False)
    valasz = kliens.predict(
        audio_path=handle_file(str(hangfajl)),
        youtube_url="",
        prompt_text=kerdes,
        api_name="/infer",
    ).strip()

    if valasz.startswith("❌"):
        raise RuntimeError(valasz)

    return valasz.removeprefix("✅ Using audio file\n\n").strip()


def recognize_song(hangfajl):

    valasz = kerdezd_flamingot(
        hangfajl,
        "Identify this recording. Reply on one line in the format Artist - Title. "
        "If you cannot identify the recording confidently, reply only Unknown.",
    )
    if valasz.rstrip(".").casefold() == "unknown":
        return "Ismeretlen"
    return valasz

def extract_features(hangfajl):
    mezok = ("tempo_bpm", "ritmus", "hangszerek", "enek", "hangulat")

    valasz = kerdezd_flamingot(
        hangfajl,
        "Describe the audible musical properties. Return one JSON object with "
        "these exact keys: tempo_bpm, ritmus, hangszerek, enek, hangulat. "
        "Use a number or null for tempo_bpm and short English strings for the "
        "other fields. Omit song titles, artist names, and genre labels.",
    )

    try:
        eleje = valasz.index("{")
        vege = valasz.rindex("}") + 1
        szoveg = valasz[eleje:vege]

        try:
            adatok = json.loads(szoveg)
        except json.JSONDecodeError:
            adatok = ast.literal_eval(szoveg)
    except (ValueError, SyntaxError) as hiba:
        raise RuntimeError(
            f"A modell válaszát nem sikerült beolvasni.\n{valasz}"
        ) from hiba

    return {mezo: adatok[mezo] for mezo in mezok if mezo in adatok}


if __name__ == "__main__":
    try:
        hangfajl = hangfajl_ellenorzese(MINTAFAJL)
        print(f"Elemzés indul: {hangfajl.name}", flush=True)
        tulajdonsagok = extract_features(hangfajl)
        print(json.dumps(tulajdonsagok, ensure_ascii=False, indent=2))
    except (FileNotFoundError, AppError, RuntimeError) as hiba:
        print(f"Az elemzés hibát jelzett. {hiba}")