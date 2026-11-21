##pip install audd
from audd import AudD
token = "9d982366f7e755b54a99017d2757a677"
audd = AudD(token)



def recognize_song(file_path):
    result = audd.recognize(file_path)
    if result:
        return result.artist, result.title
    else:
        return "nem talalhato"

    
result = recognize_song("sample.mp3")
print(result)