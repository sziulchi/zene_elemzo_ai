"""Az mp3 alapjan ismerje fel a zenet, vegye ki a nyers tulajdonsagait(stilus, utem, hangulat, hangszin stb.) utana az alapjan (chat elozmeny es mufaj megnevezese nelkul) tulajdonsagabol probalja meg felismerni a mufajt aztan adjon hasonlokat ."""
from pathlib import Path

MINTAFAJL = Path(__file__).resolve().parent / "sample" / "sample1.mp3"


def hangfajl_ellenorzese(utvonal):
    hangfajl = Path(utvonal)
    if not hangfajl.is_file():
        raise FileNotFoundError(f"A hangfájl nem található. {hangfajl}")
    return hangfajl


if __name__ == "__main__":
    hangfajl = hangfajl_ellenorzese(MINTAFAJL)
    print(f"Megtaláltam a hangfájlt: {hangfajl.name}")