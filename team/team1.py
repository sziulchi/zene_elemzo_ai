"""Dalfelismeres az AudD szolgaltatassal.

Telepites:  pip install audd
API kulcs:  AUDD_API_TOKEN kornyezeti valtozo (https://dashboard.audd.io)
"""
import os

DEFAULT_TOKEN = "9d982366f7e755b54a99017d2757a677"  # TODO: kivenni, ha publikus a repo

_audd = None


def _get_audd():
    global _audd
    if _audd is None:
        from audd import AudD
        _audd = AudD(os.environ.get("AUDD_API_TOKEN", DEFAULT_TOKEN))
    return _audd


def recognize_song(file_path):
    """Visszaadja: {"eloado": ..., "cim": ...}  vagy None, ha nem talalhato."""
    result = _get_audd().recognize(file_path)
    if result:
        return {"eloado": result.artist, "cim": result.title}
    return None


if __name__ == "__main__":
    import sys
    print(recognize_song(sys.argv[1] if len(sys.argv) > 1 else "sample/sample1.mp3")
          or "nem talalhato")
