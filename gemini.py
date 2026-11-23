import json
import os
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")

_client = None
_uploaded = {}  


class SongInfo(BaseModel):
    cim: str = Field(description="A dal cime, vagy 'ismeretlen'")
    eloado: str = Field(description="Az eloado neve, vagy 'ismeretlen'")
    album: str = Field(description="Az album neve, vagy 'ismeretlen'")
    megjelenes_eve: str = Field(description="Megjelenes eve, vagy 'ismeretlen'")
    biztonsag: float = Field(description="Mennyire biztos a felismeres, 0.0-1.0")


class Features(BaseModel):
    tempo_bpm: int = Field(description="Becsult tempo BPM-ben")
    utemmutato: str = Field(description="Pl. 4/4, 3/4, 6/8")
    hangnem: str = Field(description="Becsult hangnem, pl. 'a-moll'")
    ritmus: str = Field(description="A ritmus jellege (pl. egyenes, szinkopalt, swinges, toros)")
    hangulat: list[str] = Field(description="Hangulatot leiro melleknevek")
    energia: float = Field(description="Energiaszint 0.0-1.0")
    pozitivitas: float = Field(description="Mennyire vidam/pozitiv 0.0-1.0")
    tancolhatosag: float = Field(description="Tancolhatosag 0.0-1.0")
    hangszin: str = Field(description="A hangzas szine, textura (pl. meleg, eles, tompa, retegzett)")
    hangszerek: list[str] = Field(description="Hallhato hangszerek, hangforrasok")
    enek: str = Field(description="Enek jellege: nincs / ferfi / noi / tobb hang, es a stilusa")
    nyelv: str = Field(description="Az enek nyelve, ha van, kulonben 'nincs'")
    dinamika: str = Field(description="Hangero-valtozasok, felepites, dinamikai iv")
    produkcio: str = Field(description="Hangzas/produkcio jellege (pl. akusztikus, elektronikus, lo-fi, studio)")
    szerkezet: str = Field(description="A dal felepitese (pl. intro-verze-refren-...)")


class GenreGuess(BaseModel):
    mufaj: str
    valoszinuseg: float = Field(description="0.0-1.0")
    indoklas: str = Field(description="Mely tulajdonsagok alapjan")


class GenreResult(BaseModel):
    tippek: list[GenreGuess] = Field(description="5 tipp, csokkeno valoszinuseg szerint")


class SimilarSong(BaseModel):
    cim: str
    eloado: str
    miert_hasonlo: str


class SimilarResult(BaseModel):
    hasonlo_zenek: list[SimilarSong]


# Segedfuggvenyek

def get_client():
    """Gemini kliens (GEMINI_API_KEY kornyezeti valtozo, kulonben az api_key.txt)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and os.path.exists(KEY_FILE):
            with open(KEY_FILE, encoding="utf-8") as f:
                api_key = f.read().strip()
        if not api_key:
            raise RuntimeError("Nincs API kulcs! Tedd bele az api_key.txt fajlba a projekt mappajaban.")
        _client = genai.Client(api_key=api_key)
    return _client


def upload_audio(file_path):
    if file_path in _uploaded:
        return _uploaded[file_path]
    client = get_client()
    f = client.files.upload(file=file_path)
    # megvarjuk, amig a Gemini feldolgozza a fajlt
    while f.state and f.state.name == "PROCESSING":
        time.sleep(1)
        f = client.files.get(name=f.name)
    if f.state and f.state.name == "FAILED":
        raise RuntimeError(f"A fajl feldolgozasa nem sikerult: {file_path}")
    _uploaded[file_path] = f
    return f


def _ask(contents, schema, temperature=0.4):
    response = get_client().models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    return json.loads(response.text)


# Fo fv

def recognize_song(file_path):
    audio = upload_audio(file_path)
    prompt = (
        "Hallgasd meg ezt a hangfelvetelt, es probald felismerni, melyik dal ez. "
        "Add meg a cimet, az eloadot, az albumot es a megjelenes evet. "
        "Ha nem ismered fel biztosan, ne talalj ki semmit: irj 'ismeretlen'-t "
        "es adj alacsony biztonsagi erteket."
    )
    return _ask([prompt, audio], SongInfo, temperature=0.1)


def extract_features(file_path):
    audio = upload_audio(file_path)
    prompt = (
        "Elemezd ezt a zenet tisztan zeneileg, mint egy hangmernok. "
        "Ird le a tempot, utemet, hangnemet, ritmust, hangulatot, energiat, "
        "hangszint, hangszereket, eneket, dinamikat, produkciot es szerkezetet. "
        "FONTOS: sehol ne nevezz meg mufajt, alstilust, eloadot vagy dalcimet, "
        "csak a hallhato tulajdonsagokat ird le! Magyarul valaszolj."
    )
    return _ask([prompt, audio], Features, temperature=0.2)


def guess_genre(features):
    prompt = (
        "Egy zeneszamrol az alabbi tulajdonsagokat mertuk ki. A dalt nem ismered, "
        "csak ezeket az adatokat latod. Ezek alapjan tippeld meg a mufajat: "
        "adj 5 tippet csokkeno valoszinuseg szerint, mindegyikhez rovid indoklassal. "
        "Magyarul valaszolj.\n\n"
        + json.dumps(features, indent=2, ensure_ascii=False)
    )
    return _ask(prompt, GenreResult, temperature=0.4)


def describe_song(features, genre=None):
    prompt = (
        "Az alabbi adatok egy zeneszam kimert tulajdonsagai. Irj rola egy rovid "
        "(5-8 mondatos), kozertheto, magyar nyelvu leirast: milyen a hangulata, "
        "a tempoja, a hangzasa, a hangszerelese, es milyen helyzetben illik hallgatni.\n\n"
        + json.dumps(features, indent=2, ensure_ascii=False)
    )
    if genre:
        prompt += f"\n\nA becsult mufaj: {genre}"
    response = get_client().models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


def similar_songs(features, genre=None, n=3, exclude=None):
    prompt = (
        f"Ajanlj {n} letezo, ismert zeneszamot, amelyek hasonlitanak egy olyan "
        "zenere, aminek az alabbiak a tulajdonsagai. Csak valodi dalokat irj, "
        "mindegyiknel indokold roviden (magyarul), miert hasonlo.\n\n"
        + json.dumps(features, indent=2, ensure_ascii=False)
    )
    if genre:
        prompt += f"\n\nA becsult mufaj: {genre}"
    if exclude:
        prompt += f"\n\nEzt a dalt NE ajanld: {exclude}"
    return _ask(prompt, SimilarResult, temperature=0.7)["hasonlo_zenek"]


def analyze(file_path, n_similar=3):
    def step(msg):
        print(msg, flush=True)

    step(f"[1/6] Feltoltes a Geminire: {file_path} (modell: {MODEL}) ...")
    upload_audio(file_path)
    step("[2/6] Dal felismerese ...")
    song = recognize_song(file_path)
    step("[3/6] Tulajdonsagok kinyerese ...")
    features = extract_features(file_path)
    step("[4/6] Mufaj tippelese a tulajdonsagokbol ...")
    genres = guess_genre(features)
    top_genre = genres["tippek"][0]["mufaj"] if genres["tippek"] else None

    exclude = None
    if song["cim"] != "ismeretlen":
        exclude = f'{song["eloado"]} - {song["cim"]}'

    step("[5/6] Leiras irasa ...")
    leiras = describe_song(features, top_genre)
    step("[6/6] Hasonlo zenek keresese ...")
    hasonlok = similar_songs(features, top_genre, n_similar, exclude)

    return {
        "dal": song,
        "tulajdonsagok": features,
        "mufaj_tippek": genres["tippek"],
        "leiras": leiras,
        "hasonlo_zenek": hasonlok,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample/sample1.mp3"
    result = analyze(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
