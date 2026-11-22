"""Az Essentia + Groq + AudD alapu valtozat.

A modulok csak akkor toltodnek be, amikor tenylegesen hasznaljuk oket,
igy a hianyzo csomagok (pl. essentia Windowson) nem okoznak hibat importkor.
"""


def recognize_song(file_path):
    from .team1 import recognize_song as _rs
    return _rs(file_path)


def analyze(file_path):
    from .team2 import analyze as _an
    return _an(file_path)


def ask_llm(result):
    from .team3 import ask_llm as _ask
    return _ask(result)
