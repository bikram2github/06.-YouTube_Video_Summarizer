
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()

@st.cache_resource
def get_transcript_api():
    return YouTubeTranscriptApi()

yt_api = get_transcript_api()

st.title("YouTube Video Summarizer")
st.subheader("Summarize YouTube videos using Groq LLM")



api_key = st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def get_groq_model(api_key: str):
    return ChatGroq(api_key=api_key, model="openai/gpt-oss-120b", temperature=0.6)


prompt=ChatPromptTemplate.from_messages([
    ("system","""You are an intelligent assistant trained to summarize YouTube video transcripts.

Your goal is to create a clear, structured, and detailed summary that captures all important ideas, insights, and examples.
Avoid copying phrases directly from the transcript — use your own words for clarity and flow.

Follow this structure in your answer:
1. **Overview:** Briefly describe what the video is about and its main purpose.
2. **Main Points:** List the key ideas, arguments, or sections discussed in the video.
3. **Supporting Details:** Include important examples.
4. **Takeaways:** Highlight the main lessons, conclusions, or insights the viewer should remember.

Keep the summary concise and easy to read.
Ignore timestamps, filler words, or irrelevant parts of the transcript.
"""),
    ("user","{text}")
])

prompt_hindi=ChatPromptTemplate.from_messages([
    ("system","""You are a professional translator. 
Translate the following Hindi text into clear, natural English. 
Preserve the original meaning and tone.
"""),
    ("user","{text}")
])



url= st.text_input("Enter YouTube Video URL",label_visibility="collapsed", placeholder="https://www.youtube.com/watch?v=_Kj275Q4sy4")


@st.cache_data(show_spinner=False)
def summarize_hindi(text):
    if len(text) > 200:
        return None
    else:
        try:
            llm = get_groq_model(api_key)
            chain = prompt_hindi | llm | parser | prompt | llm | parser
            response = chain.invoke({"text": text})
            return response
        except Exception as e:
            return None


@st.cache_data(show_spinner=False)
def summarize_english(text):
    if len(text) > 200:
        return None
    else:
        try:
            llm = get_groq_model(api_key)
            chain = prompt | llm | parser
            response = chain.invoke({"text": text})
            return response
        except Exception as e:
            return None


@st.cache_data(show_spinner=False)
def extract(url):
    try:
        video_id= url.split("=")[1]

        fetched = yt_api.fetch(video_id, languages=["en","hi"]) 
        raw = fetched.to_raw_data()               
        full_yt_text = " ".join(d["text"] for d in raw)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "।", " ", ""]
        )
        texts = text_splitter.split_text(full_yt_text)       
        return texts

    except Exception as e:
        return None



@st.cache_data(show_spinner=False)
def language_check(text):
    text=text[0]
    eng=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    if any(char in text.lower() for char in eng):
        return True
    else:
        return False


def summarize(text):
    try:
        if language_check(text):
            summary = summarize_english(text)
            return summary
        else:
            summary = summarize_hindi(text)
            return summary
    except Exception as e:
        return st.error(f"An error occurred: {e}")





if st.button("Summarize"):
    if not api_key.strip() or not url.strip():
        st.error("Please enter your Groq API Key and YouTube Video URL.")
    elif "youtube" not in url:
        st.error("Please enter a valid YouTube Video URL.")
    else:
        with st.spinner("Extracting video content..."):
            yt_text = extract(url)

            if yt_text is None:
                st.error("An error occurred while fetching the transcript. Please ensure the video has subtitles available.")

            else:
                with st.spinner("Generating summary..."):
                    summary = summarize(yt_text)
                    st.subheader("Video Thumbnail")
                    video_id = url.split("=")[1]
                    st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg", width="content")
                    st.markdown("### Video Link")
                    st.link_button("Direct Video Link", url)


                    if summary is None:
                        st.subheader("Video Summary")
                        st.error("Input text too long for summarization.")

                    else:
                        st.subheader("Video Summary")
                        st.success(summary)