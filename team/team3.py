
import json
import os

SYSTEM_PROMPT = (
    "Egy zenenek a tulajdonsagait kivettunk Essentia-val. Kitalalod a zene mufajat? "
    "Miutan tettel 5 tippet ajanlj 3 a tulajdonsagabaibol adodoan zeneket."
)


def ask_llm(result, model="openai/gpt-oss-120b"):
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Nincs beallitva a GROQ_API_KEY kornyezeti valtozo!")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(result, indent=2, ensure_ascii=False)},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    from team.team2 import analyze

    print(ask_llm(analyze(sys.argv[1] if len(sys.argv) > 1 else "sample/sample1.mp3")))
