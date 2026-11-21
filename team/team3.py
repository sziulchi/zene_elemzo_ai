#pip install -q openai
from openai import OpenAI
from google.colab import userdata

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=userdata.get("music_analyzer"))   # a kulcsot a Colab Titkok közé mentsétek


system_prompt = "Egy zenenek a tulajdonsagait kivettunk Essentia-val. Kitalalod a zene mufajat? Miutan tettel 5 tippet ajanlj 3 a tulajdonsagabaibol adodoan zeneket."
user_prompt = json.dumps(result, indent=2, ensure_ascii=False)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",          # vagy: "openai/gpt-oss-120b"
    messages=[{"role": "user", "content": system_prompt},
              {"role": "user", "content": user_prompt}
             ]
)
print(response.choices[0].message.content)