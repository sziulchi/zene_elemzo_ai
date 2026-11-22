"""Zene elemzo AI - belepesi pont.

Hasznalat:
    python main.py                          -> Gemini, sample/sample1.mp3
    python main.py zene.mp3                 -> Gemini, sajat fajl
    python main.py zene.mp3 --mod essentia  -> AudD + Essentia + Groq (csapat valtozat)
    python main.py zene.mp3 --json          -> nyers JSON kimenet
"""
import argparse
import json
import os
import sys

# hogy barhonnan inditva megtalalja a gemini.py-t es a team csomagot
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

# Windows konzolon az ekezetes karakterek miatt
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def print_gemini(result):
    dal = result["dal"]
    print("\n=== FELISMERT DAL ===")
    if dal["cim"] == "ismeretlen":
        print("Nem sikerult biztosan felismerni.")
    else:
        print(f'{dal["eloado"]} - {dal["cim"]}  ({dal["album"]}, {dal["megjelenes_eve"]})'
              f'  [biztonsag: {dal["biztonsag"]:.0%}]')

    print("\n=== TULAJDONSAGOK ===")
    for k, v in result["tulajdonsagok"].items():
        if isinstance(v, list):
            v = ", ".join(v)
        print(f"  {k:14}: {v}")

    print("\n=== MUFAJ TIPPEK (csak a tulajdonsagok alapjan) ===")
    for i, t in enumerate(result["mufaj_tippek"], 1):
        print(f'  {i}. {t["mufaj"]} ({t["valoszinuseg"]:.0%}) - {t["indoklas"]}')

    print("\n=== LEIRAS ===")
    print(result["leiras"])

    print("\n=== HASONLO ZENEK ===")
    for s in result["hasonlo_zenek"]:
        print(f'  - {s["eloado"]} - {s["cim"]}: {s["miert_hasonlo"]}')


def run_gemini(path, as_json):
    import gemini
    result = gemini.analyze(path)
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_gemini(result)


def run_essentia(path):
    import team
    print("=== FELISMERES (AudD) ===")
    print(team.recognize_song(path) or "nem talalhato")
    print("\n=== TULAJDONSAGOK (Essentia) ===")
    features = team.analyze(path)
    print(json.dumps(features, indent=2, ensure_ascii=False))
    print("\n=== MUFAJ + AJANLAS (Groq) ===")
    print(team.ask_llm(features))


def main():
    parser = argparse.ArgumentParser(description="Zene elemzo AI")
    parser.add_argument("file", nargs="?", default="sample/sample1.mp3", help="mp3 fajl")
    parser.add_argument("--mod", choices=["gemini", "essentia"], default="gemini")
    parser.add_argument("--json", action="store_true", help="nyers JSON kimenet")
    args = parser.parse_args()
    print(f"Zene elemzo AI indul ({args.mod} mod) ...", flush=True)

    if not os.path.exists(args.file):
        sys.exit(f"Nincs ilyen fajl: {args.file}")

    try:
        if args.mod == "gemini":
            run_gemini(args.file, args.json)
        else:
            run_essentia(args.file)
    except ModuleNotFoundError as e:
        sys.exit(f"Hianyzo csomag: {e.name}  ->  pip install -r requirements.txt")
    except RuntimeError as e:
        sys.exit(f"Hiba: {e}")


if __name__ == "__main__":
    main()
