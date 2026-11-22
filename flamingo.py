"""Az mp3 alapjan ismerje fel a zenet, vegye ki a nyers tulajdonsagait(stilus, utem, hangulat, hangszin stb.) utana az alapjan (chat elozmeny es mufaj megnevezese nelkul) tulajdonsagabol probalja meg felismerni a mufajt aztan adjon hasonlokat ."""
import ast
import os
import json
from functools import cache
from pathlib import Path

import librosa
from argostranslate.translate import translate
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MINTAFAJL = Path(__file__).resolve().parent / "sample" / "sample1.mp3"


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
            "A modell egy része lemezre került. CUDA-s gépen próbáld a "
            "FLAMINGO_4BIT=1 beállítást, vagy futtasd több memóriával."
        )

    return processor, model


def kerdezd_flamingot(kerdes, hangfajl=None):
    processor, model = modell_betoltese()
    tartalom = [{"type": "text", "text": kerdes}]

    if hangfajl is not None:
        hang, _ = librosa.load(
            str(hangfajl),
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
            duration=25,
        )
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
        max_new_tokens=500,
        do_sample=False,
        repetition_penalty=1.2,
    )
    uj_tokenek = kimenet[:, bemenet["input_ids"].shape[1]:]
    return processor.batch_decode(
        uj_tokenek,
        skip_special_tokens=True,
    )[0].strip()


def szotar_beolvasasa(valasz):
    try:
        eleje = valasz.index("{")
        vege = valasz.rindex("}") + 1
        szoveg = valasz[eleje:vege]

        try:
            adatok = json.loads(szoveg)
        except json.JSONDecodeError:
            adatok = ast.literal_eval(szoveg)

        if not isinstance(adatok, dict):
            raise ValueError("A válasz formátuma eltér a várt szótártól.")

        return adatok
    except (ValueError, SyntaxError) as hiba:
        raise RuntimeError(
            f"A modell válasza nem olvasható be.\n{valasz}"
        ) from hiba


def recognize_song(hangfajl):
    valasz = kerdezd_flamingot(
        "Identify the recording. Reply on one line as Artist - Title. "
        "If you cannot identify it confidently, reply only Unknown.",
        hangfajl,
    )
    if valasz.rstrip(".").casefold() == "unknown":
        return "Ismeretlen"
    return valasz


def extract_features(hangfajl):
    mezok = ("tempo_bpm", "ritmus", "hangszerek", "hangulat")
    valasz = kerdezd_flamingot(
        f"Describe the audible music as one JSON object with keys {', '.join(mezok)}. "
        "Use a number or null for tempo_bpm and English words for the other fields. "
        "List up to three audible instruments and up to three mood adjectives. "
        "Omit genre labels, titles and artist names. Return only the JSON object.",
        hangfajl,
    )
    adatok = szotar_beolvasasa(valasz)
    return {mezo: adatok[mezo] for mezo in mezok if mezo in adatok}


DALLISTA = {
    "1": ("AURORA", "Cure For Me", "electropop"),
    "2": ("CHVRCHES", "The Mother We Share", "synth-pop"),
    "3": ("M83", "Midnight City", "dreamy synth-pop"),
    "4": ("The Weeknd", "Blinding Lights", "upbeat synth-pop"),
    "5": ("Daft Punk", "One More Time", "dance and house"),
    "6": ("Paramore", "Hard Times", "pop rock and new wave"),
    "7": ("Mr. Fingers", "Can You Feel It", "deep house"),
    "8": ("Nora En Pure", "Come With Me", "deep house"),
    "9": ("Kerri Chandler", "Bar A Thym", "deep house"),
}


def mufaj_es_ajanlas(tulajdonsagok):
    jellemzok = json.dumps(tulajdonsagok, ensure_ascii=False)
    eredmeny = szotar_beolvasasa(kerdezd_flamingot(
        "Use only the musical properties below. Return one JSON object "
        'with "mufaj" for the most likely genre and "leiras" '
        "for a short English description of the audible properties."
        "\n\n" + jellemzok
    ))

    mufaj = str(eredmeny["mufaj"]).casefold().replace("-", " ")
    house_mufaj = "house" in mufaj or "mély ház" in mufaj
    jelolt_dalok = DALLISTA

    if house_mufaj:
        jelolt_dalok = {
            azonosito: dal
            for azonosito, dal in DALLISTA.items()
            if dal[2] == "deep house"
        }

    print(f"Választható dalok: {', '.join(jelolt_dalok)}", flush=True)
    jeloltek = "\n".join(
        f"{azonosito}. {eloado} - {cim} ({stilus})"
        for azonosito, (eloado, cim, stilus) in jelolt_dalok.items()
    )
    nyers_valasz = kerdezd_flamingot(
        "Pick up to three distinct tracks from the supplied catalogue. "
        'Return only JSON with an "ajanlasok" list. Each item must contain '
        '"id", a number from the supplied catalogue, and "indoklas", '
        "a short English reason based on the musical properties."
        f"\n\nMusical properties:\n{jellemzok}"
        f"\n\nCatalogue:\n{jeloltek}"
    )
    try:
        ajanlasok = szotar_beolvasasa(nyers_valasz).get("ajanlasok")
    except RuntimeError:
        ajanlasok = []

    hasonlo_zenek = []
    hasznalt = set()
    if isinstance(ajanlasok, list):
        for ajanlas in ajanlasok:
            if not isinstance(ajanlas, dict):
                continue
            azonosito = str(ajanlas.get("id", "")).strip()
            if azonosito in jelolt_dalok and azonosito not in hasznalt:
                eloado, cim, _ = jelolt_dalok[azonosito]
                hasonlo_zenek.append({
                    "eloado": eloado,
                    "cim": cim,
                    "indoklas": ajanlas.get("indoklas") or "Selected by the model.",
                })
                hasznalt.add(azonosito)

    if not hasonlo_zenek and house_mufaj:
        for eloado, cim, _ in jelolt_dalok.values():
            hasonlo_zenek.append({
                "eloado": eloado,
                "cim": cim,
                "indoklas": "This track matches the estimated house genre.",
            })

    eredmeny["hasonlo_zenek"] = hasonlo_zenek[:3]
    return eredmeny


def magyarra(ertek):
    if isinstance(ertek, dict):
        return "\n".join(
            f"{magyarra(kulcs.replace('_', ' '))}: {magyarra(tartalom)}"
            for kulcs, tartalom in ertek.items()
        )

    if isinstance(ertek, list):
        return ", ".join(magyarra(elem) for elem in ertek)

    if ertek is None:
        return "Nincs adat"

    if isinstance(ertek, str) and ertek.strip():
        return translate(ertek, "en", "hu")

    return str(ertek)


if __name__ == "__main__":
    if not MINTAFAJL.is_file():
        raise FileNotFoundError(f"A hangfájl nem található. {MINTAFAJL}")

    print(f"Elemzés indul: {MINTAFAJL.name}", flush=True)
    print("Előadó és dalcím:", recognize_song(MINTAFAJL), flush=True)

    print("Zenei jellemzők kinyerése…", flush=True)
    tulajdonsagok = extract_features(MINTAFAJL)

    print("Műfajbecslés és hasonló zenék keresése…", flush=True)
    eredmeny = mufaj_es_ajanlas(tulajdonsagok)

    print("\nBecsült műfaj:", magyarra(eredmeny["mufaj"]))
    print("\nA zene jellemzői")
    print(magyarra(eredmeny["leiras"]))

    print("\nHasonló zenék")
    for dal in eredmeny["hasonlo_zenek"]:
        print(f"- {dal['eloado']} – {dal['cim']}. {magyarra(dal['indoklas'])}")