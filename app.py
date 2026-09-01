import streamlit as st
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer,util
import whisper
import json
import streamlit as st


st.title("VIDEO SEARCH FOR NLP")
@st.cache_resource

# Audio to text transcribe
def load_whisper_model():
    return whisper.load_model("small")
# result =text,segment(start,end,text[start:end],seek etc)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

whisper_model=load_whisper_model()
text_model=load_model()

#temporary store the uploaded video
video_file=st.file_uploader(" upload your video",type=["mp4"])
if video_file is not None:
    if "data" not in st.session_state:   # process(transcribe) video only once
        path="temp.mp4"
        with open(path,"wb") as f:
            f.write(video_file.read())

#audio extract with moviepy as videofileclip
        video=VideoFileClip(path)
        audio_path="temp_audio.wav"
        audio=video.audio
        audio.write_audiofile(audio_path)
        video.close()


# audio to text using whisper
        result=whisper_model.transcribe(audio_path)
        segments=result["segments"]

        data = [
        {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
        for seg in segments
        ]

# dump() json file for search (save in form of json)
        with open("transcript.json","w",encoding="utf-8") as f:
            data=json.dump(data,f,indent=2)


        with open("transcript.json","r",encoding="utf-8") as f:
            data=json.load(f)
# embeddings
# extract transcript
        st.session_state.data=data
        sentence=[seg["text"] for seg in data]

#embediings
        st.session_state.embeddings=text_model.encode(sentence)
        st.session_state.video_file=video_file
    else:
        video_file=st.session_state.video_file
        embeddings=st.session_state.embeddings
        data=st.session_state.data


# user query
query=st.text_input("What do you want to search? ")

if st.button("search"):
    if query:
        query_emb=text_model.encode(query)

       # calculate smilarity
        scores=util.cos_sim(query_emb,st.session_state.embeddings)[0]

        min_score=0.5    #threshold
        top_res=scores.argsort(descending=True)[:5]
        found=False
        for idx in top_res:
           idx=int(idx)
           score=float(scores[idx])
           if score>=min_score:
                found=True
                print("text:",data[idx]["text"])
                print("start:",data[idx]["start"])
                print("end:",data[idx]["end"])
                print("score",score)
                if video_file:
                   st.video(video_file,start_time=int(data[idx]["start"]))
                st.write("-------")
        if not found: 
            st.write("no relevent feature matches ")  
else:
    st.write("type query")            

