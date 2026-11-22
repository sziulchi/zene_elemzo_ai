"""Az mp3 alapjan ismerje fel a zenet, vegye ki a nyers tulajdonsagait(stilus, utem, hangulat, hangszin stb.) utana az alapjan (chat elozmeny es mufaj megnevezese nelkul) tulajdonsagabol probalja meg felismerni a mufajt aztan adjon hasonlokat ."""
import argparse
import asyncio
import ast
import json
import os
import re
from functools import cache
from pathlib import Path
from unicodedata import normalize
from urllib.parse import urlencode
from urllib.request import Request, getproxies, urlopen

import librosa
from argostranslate.translate import translate
from mutagen import File as MutagenFile, MutagenError
from shazamio import Shazam
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration


MODEL_ID = "nvidia/music-flamingo-2601-hf"
MINTAFAJL = Path(__file__).resolve().parent / "sample" / "sample1.mp3"

MEZOK = {
    "tempo_bpm": "Becsült tempó",
    "ritmus": "Ritmus",
    "hangszerek": "Hangszerek",
    "hangszin": "Hangszín",
    "hangulat": "Hangulat",
}

ZENEI_KIFEJEZESEK = {
    "fast": "gyors",
    "slow": "lassú",
    "steady": "egyenletes",
    "uplifting": "felemelő",
    "hopeful": "reményteli",
    "melancholic": "melankolikus",
    "electronic drums": "elektronikus dobok",
    "synth pads": "kitartott szintetizátorhangok",
}


@cache
def modell_betoltese():
    print("A Music Flamingo betöltése indul.", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    beallitasok = {"device_map": "auto", "dtype": "auto"}

    if os.environ.get("FLAMINGO_4BIT") == "1":
        import torch
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("A 4 bites módhoz CUDA-s PyTorch szükséges.")

        beallitasok = {
            "device_map": {"": 0},
            "dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        }

    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        MODEL_ID, **beallitasok
    )
    if "disk" in getattr(model, "hf_device_map", {}).values():
        raise RuntimeError(
            "A modell egy része lemezre került. Használj több memóriát, "
            "vagy CUDA-s gépen próbáld a FLAMINGO_4BIT=1 beállítást."
        )

    return processor, model


def kerdezd_flamingot(kerdes, hang=None, max_token=200):
    processor, model = modell_betoltese()
    tartalom = [{"type": "text", "text": kerdes}]
    if hang is not None:
        tartalom.append({"type": "audio", "audio": hang})

    bemenet = processor.apply_chat_template(
        [{"role": "user", "content": tartalom}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if "input_features" in bemenet:
        bemenet["input_features"] = bemenet["input_features"].to(model.dtype)

    kimenet = model.generate(
        **bemenet,
        max_new_tokens=max_token,
        do_sample=False,
        repetition_penalty=1.2,
    )
    uj_tokenek = kimenet[:, bemenet["input_ids"].shape[1]:]
    return processor.batch_decode(
        uj_tokenek, skip_special_tokens=True
    )[0].strip()


def szotar_beolvasasa(valasz):
    try:
        szoveg = valasz[valasz.index("{"):valasz.rindex("}") + 1]
        try:
            adatok = json.loads(szoveg)
        except json.JSONDecodeError:
            adatok = ast.literal_eval(szoveg)

        if isinstance(adatok, dict):
            return adatok
    except (ValueError, SyntaxError):
        pass

    raise RuntimeError(f"A modell nem adott olvasható szótárt.\n{valasz}")


async def shazam_felismeres(hangfajl):
    async with Shazam() as shazam:
        return await asyncio.wait_for(
            shazam.recognize(
                str(hangfajl),
                proxy=getproxies().get("https"),
            ),
            timeout=20,
        )


def recognize_song(hangfajl):
    try:
        talalat = asyncio.run(shazam_felismeres(hangfajl))
        dal = talalat.get("track") or {}
        eloado = dal.get("subtitle")
        cim = dal.get("title")

        if eloado and cim:
            return f"{eloado} – {cim} (hanglenyomat alapján)"
    except Exception as hiba:
        print(f"A hanglenyomatos keresés megállt ({type(hiba).__name__}).")

    try:
        metadata = MutagenFile(str(hangfajl), easy=True)
    except (MutagenError, OSError):
        metadata = None

    if metadata is not None:
        eloado = " / ".join(metadata.get("artist", [])).strip()
        cim = " / ".join(metadata.get("title", [])).strip()
        if eloado and cim:
            return f"{eloado} – {cim} (a fájl címkéje)"

    return "Ismeretlen"


def extract_features(hang):
    valasz = kerdezd_flamingot(
        "Describe only audible musical properties in English. "
        'Return one JSON object with keys "tempo_bpm", "ritmus", '
        '"hangszerek", "hangszin", "hangulat". '
        "Use a number or null for tempo_bpm. Name up to three instruments "
        "and up to three mood adjectives. Use short values. "
        "Omit all genre labels, song titles and artist names.",
        hang,
    )
    adatok = szotar_beolvasasa(valasz)
    jellemzok = {mezo: adatok[mezo] for mezo in MEZOK if mezo in adatok}

    if not jellemzok:
        raise RuntimeError("A modell válaszából hiányoznak a zenei jellemzők.")

    return jellemzok


def guess_genre(jellemzok):
    valasz = kerdezd_flamingot(
        'Use only these musical properties. Return one JSON object with a '
        'single "mufaj" field containing the most likely genre in English.'
        "\n\n" + json.dumps(jellemzok, ensure_ascii=False),
        max_token=80,
    )
    eredmeny = szotar_beolvasasa(valasz)
    mufaj = eredmeny.get("mufaj")
    return mufaj.strip() if isinstance(mufaj, str) and mufaj.strip() else "Ismeretlen"


def keresesi_nev(szoveg):
    betuk = normalize("NFKD", szoveg.casefold())
    return "".join(betu for betu in betuk if betu.isalnum())


def itunes_dalok(eloado):
    parameterek = urlencode({
        "term": eloado,
        "media": "music",
        "entity": "song",
        "attribute": "artistTerm",
        "country": "US",
        "limit": 20,
    })
    keres = Request(
        f"https://itunes.apple.com/search?{parameterek}",
        headers={
            "Accept": "application/json",
            "User-Agent": "FlamingoMusicProject/1.0",
        },
    )
    with urlopen(keres, timeout=8) as valasz:
        talalatok = json.load(valasz)["results"]

    dalok = {}
    for talalat in talalatok:
        nev = talalat.get("artistName", "")
        cim = talalat.get("trackName", "")
        if cim and keresesi_nev(nev) == keresesi_nev(eloado):
            dalok.setdefault(keresesi_nev(cim), (
                nev,
                cim,
                talalat.get("primaryGenreName") or "műfajadat nélkül",
            ))

    return list(dalok.values())[:3]


def musicbrainz_dalok(mufaj):
    mufajok = [
        " ".join(re.findall(r"[\w-]+", resz.casefold()))
        for resz in re.split(r"[/,;]", mufaj)
    ]
    mufajok = [resz for resz in mufajok if resz]
    if not mufajok:
        return []

    query = " OR ".join(f'tag:"{resz}"' for resz in mufajok)
    parameterek = urlencode({"query": query, "fmt": "json", "limit": 15})
    keres = Request(
        f"https://musicbrainz.org/ws/2/recording/?{parameterek}",
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "FlamingoMusicProject/1.0 "
                "(https://github.com/sziulchi/zene_elemzo_ai)"
            ),
        },
    )

    try:
        with urlopen(keres, timeout=8) as valasz:
            felvetelek = json.load(valasz)["recordings"]
    except (OSError, ValueError, KeyError):
        print("A MusicBrainz műfajkeresése most nem érhető el.", flush=True)
        return []

    dalok = {}
    for felvetel in felvetelek:
        cim = felvetel.get("title", "")
        eloado = "".join(
            resz["name"] + resz.get("joinphrase", "")
            for resz in felvetel.get("artist-credit", [])
            if resz.get("name")
        )
        cimkek = {tag["name"].casefold() for tag in felvetel.get("tags", [])}
        egyezes = next((cimke for cimke in mufajok if cimke in cimkek), None)

        if cim and eloado and egyezes:
            kulcs = (keresesi_nev(eloado), keresesi_nev(cim))
            dalok.setdefault(kulcs, (eloado, cim, egyezes))

    return list(dalok.values())[:3]

def itunes_mufaj_dalok(mufaj):
    def lekeres(vegpont, parameterek):
        query = urlencode({"country": "US", **parameterek})
        keres = Request(
            f"https://itunes.apple.com/{vegpont}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": "FlamingoMusicProject/1.0",
            },
        )
        with urlopen(keres, timeout=8) as valasz:
            return json.load(valasz)["results"]

    print("Műfaj szerinti válogatások keresése az iTunes-ban…", flush=True)

    try:
        for resz in re.split(r"[/,;]", mufaj):
            cimke = resz.strip()
            keresett = keresesi_nev(cimke)
            if not keresett:
                continue

            albumok = lekeres("search", {
                "term": cimke,
                "media": "music",
                "entity": "album",
                "limit": 15,
            })

            for album in albumok:
                if (
                    album.get("artistName", "").casefold() != "various artists"
                    or keresett not in keresesi_nev(album.get("collectionName", ""))
                    or not album.get("collectionId")
                ):
                    continue

                szamok = lekeres("lookup", {
                    "id": album["collectionId"],
                    "entity": "song",
                    "limit": 30,
                })

                dalok = {}
                for szam in szamok:
                    eloado = szam.get("artistName")
                    cim = szam.get("trackName")
                    if szam.get("kind") == "song" and eloado and cim:
                        kulcs = (keresesi_nev(eloado), keresesi_nev(cim))
                        dalok.setdefault(
                            kulcs, (eloado, cim, f"{cimke}, válogatásból")
                        )

                if dalok:
                    return list(dalok.values())[:3]

    except (OSError, ValueError, KeyError) as hiba:
        print(f"Az iTunes műfajkeresése nem sikerült ({hiba}).", flush=True)

    return []

def similar_songs(jellemzok, mufaj):
    alap = json.dumps(
        {"mufaj": mufaj, "jellemzok": jellemzok},
        ensure_ascii=False,
    )
    valasz = kerdezd_flamingot(
        "Based on this music analysis, suggest five real, established artists "
        "who have released music with a similar sound. Return only a JSON object "
        'with an "eloadok" list of artist names. Use their official names.'
        f"\n\n{alap}",
        max_token=150,
    )
    try:
        eloadok = szotar_beolvasasa(valasz).get("eloadok")
    except RuntimeError:
        eloadok = []

    if not isinstance(eloadok, list):
        eloadok = []

    nevek = {}
    for eloado in eloadok:
        if isinstance(eloado, str) and eloado.strip():
            nevek.setdefault(keresesi_nev(eloado), eloado.strip())

    jeloltek = []
    tartalek = []
    try:
        for eloado in list(nevek.values())[:5]:
            dalok = itunes_dalok(eloado)
            if dalok:
                jeloltek.extend(dalok)
                tartalek.append(dalok[0])
    except (OSError, ValueError, KeyError):
        print("A zenekatalógus lekérdezése közben hiba történt.", flush=True)

    if not jeloltek:
        print("Ajánlások keresése műfaj alapján…", flush=True)
        return musicbrainz_dalok(mufaj) or itunes_mufaj_dalok(mufaj)

    katalogus = "\n".join(
        f"{sorszam}. {eloado} - {cim} ({stilus})"
        for sorszam, (eloado, cim, stilus) in enumerate(jeloltek, start=1)
    )
    valasz = kerdezd_flamingot(
        "Rank up to three tracks from the search results by musical similarity. "
        "Reply only with comma-separated numbers from the results."
        f"\n\nMusic analysis:\n{alap}\n\nSearch results:\n{katalogus}",
        max_token=40,
    )

    if not re.fullmatch(r"\d{1,2}(?:\s*,\s*\d{1,2}){0,2}", valasz.strip()):
        return tartalek[:3]

    sorszamok = dict.fromkeys(int(szam) for szam in re.findall(r"\d+", valasz))
    ajanlasok = [
        jeloltek[szam - 1]
        for szam in sorszamok
        if 1 <= szam <= len(jeloltek)
    ]
    return ajanlasok or tartalek[:3]


def magyarra(ertek):
    if isinstance(ertek, list):
        return ", ".join(magyarra(elem) for elem in ertek) or "Ismeretlen"
    if ertek is None:
        return "Ismeretlen"
    if not isinstance(ertek, str):
        return str(ertek)

    szoveg = ertek.strip()
    if not szoveg:
        return "Ismeretlen"

    normalizalt = szoveg.casefold()
    if "four-on-the-floor" in normalizalt or "4-on-the-floor" in normalizalt:
        return "egyenletes lábdob minden negyedre"
    if normalizalt in ZENEI_KIFEJEZESEK:
        return ZENEI_KIFEJEZESEK[normalizalt]

    return translate(szoveg, "en", "hu")


def main():
    parser = argparse.ArgumentParser(description="Helyi Music Flamingo zeneelemző")
    parser.add_argument("hangfajl", type=Path, nargs="?", default=MINTAFAJL)
    args = parser.parse_args()

    if not args.hangfajl.is_file():
        parser.exit(1, f"A hangfájl nem található. {args.hangfajl}\n")

    print(f"Elemzés indul: {args.hangfajl.name} (első 25 másodperc)", flush=True)

    try:
        print("Előadó és dalcím:", recognize_song(args.hangfajl), flush=True)
        processor, _ = modell_betoltese()
        hang, _ = librosa.load(
            str(args.hangfajl),
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
            duration=25,
        )

        print("Zenei jellemzők kinyerése…", flush=True)
        jellemzok = extract_features(hang)
        print("Műfajbecslés és ajánlás…", flush=True)
        mufaj = guess_genre(jellemzok)
        ajanlasok = similar_songs(jellemzok, mufaj)

        print(f"\nBecsült műfaj: {mufaj}")
        print("\nA zene jellemzői")
        for mezo, felirat in MEZOK.items():
            if mezo in jellemzok:
                ertek = jellemzok[mezo]
                szoveg = (
                    f"{ertek} BPM"
                    if mezo == "tempo_bpm" and isinstance(ertek, (int, float))
                    else magyarra(ertek)
                )
                print(f"- {felirat}: {szoveg}")

        print("\nHasonló zenék")
        if ajanlasok:
            for eloado, cim, stilus in ajanlasok:
                print(f"- {eloado} – {cim} ({stilus})")
        else:
            print("A katalógusban ehhez az elemzéshez nincs használható ajánlás.")
    except (OSError, RuntimeError) as hiba:
        parser.exit(1, f"Az elemzés megállt. {hiba}\n")


if __name__ == "__main__":
    main()