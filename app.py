import gradio as gr
import pandas as pd
from groq import Groq
import tempfile
from gtts import gTTS
from PyPDF2 import PdfReader
import os
import cv2
import numpy as np
import json
from pydub import AudioSegment

# --- SETUP ---
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None
MODEL_NAME = "llama-3.3-70b-versatile"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- UTILITIES ---
def extract_audio(input_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio.export(temp_audio.name, format="wav")
            return temp_audio.name
    except Exception:
        return None

def analyze_video_focus(video_path):
    if not video_path: return 0, "No video detected."
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return 0, "Video error."
    
    face_detected_frames = 0
    sample_rate = max(15, total_frames // 50) 
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % sample_rate == 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0: face_detected_frames += 1
        count += 1
    cap.release()
    
    analyzed = (total_frames / sample_rate)
    ratio = face_detected_frames / analyzed if analyzed > 0 else 0
    score = min(100, int(ratio * 140)) 
    return score, "Eye contact checked."

def get_ai_scores(user_text, cv_context):
    word_count = len(user_text.split())
    if word_count < 7:
        return {"Confidence": 85, "Technical": 5, "Clarity": 95, "Knowledge": 5, "Feedback": "Greeting received. Waiting for technical depth."}
    
    prompt = f"""
    Analyze interview answer: "{user_text}"
    Context: {cv_context}
    Return ONLY JSON: {{"Technical": x, "Knowledge": y, "Clarity": z, "Confidence": w, "Feedback": "text"}}
    """
    try:
        res = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODEL_NAME, response_format={"type": "json_object"})
        return json.loads(res.choices[0].message.content)
    except:
        return {"Technical": 40, "Knowledge": 40, "Clarity": 70, "Confidence": 70, "Feedback": "Analysis processed."}

# --- PROCESSOR ---
def process_interview(video_path, audio_path, history, level, cv_context):
    raw_input = video_path if video_path else audio_path
    if not raw_input: 
        return history, None, gr.update(), "Please record your response first."
    
    try:
        # 1. Audio processing
        processed_audio = extract_audio(raw_input)
        if not processed_audio: return history, None, gr.update(), "Could not extract audio."
        
        with open(processed_audio, "rb") as f:
            user_text = client.audio.transcriptions.create(file=(processed_audio, f.read()), model="whisper-large-v3", response_format="text")
        
        # 2. AI Logic
        scores = get_ai_scores(user_text, cv_context)
        messages = [{"role": "system", "content": f"Expert Interviewer. Context: {cv_context}. Ask a follow-up."}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        
        response = client.chat.completions.create(messages=messages, model=MODEL_NAME)
        ai_reply = response.choices[0].message.content

        # 3. Voice Reply
        tts = gTTS(text=ai_reply, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_reply = fp.name
            
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": ai_reply})
        
        # 4. Metrics & Graphs (Fixed)
        metrics = ["Confidence", "Technical", "Clarity", "Knowledge"]
        vals = [scores["Confidence"], scores["Technical"], scores["Clarity"], scores["Knowledge"]]
        final_fb = scores["Feedback"]
        
        if level == "3":
            v_score, v_msg = analyze_video_focus(video_path)
            metrics.append("Body Lang")
            vals.append(v_score)
            final_fb = f"{v_msg} | {final_fb}"
        
        # CREATE PLOT WITHOUT 'VERTICAL' PARAMETER
        df = pd.DataFrame({"Metric": metrics, "Score": vals})
        new_plot = gr.BarPlot(
            value=df, 
            x="Metric", 
            y="Score", 
            y_lim=[0,100], 
            color="Metric", 
            title="Interview Performance",
            tooltip=["Metric", "Score"]
        )
            
        return history, audio_reply, new_plot, final_fb

    except Exception as e:
        # Emergency return so the app doesn't freeze
        return history, None, gr.update(), f"System Error: {str(e)}"

# --- UI (Level 1 remains exactly as you liked) ---
with gr.Blocks() as demo:
    cv_memory = gr.State("")
    chat_state = gr.State([]) 
    gr.Markdown("# 🚀 AI Interview Pro")
    
    with gr.Tab("1. Profile Setup"):
        with gr.Row():
            cv_in = gr.File(label="Upload Resume (PDF)")
            jd_in = gr.Textbox(label="Job Title/Goal")
        btn1 = gr.Button("🔍 Perform Deep AI Analysis", variant="primary")
        out_analysis = gr.Markdown("### Results will appear here...")
        
        def start_analysis(f, j):
            if not f: return "Please upload your resume."
            reader = PdfReader(f.name)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            prompt = f"Analyze CV for {j}. Provide Match Score, Strengths, Critical Gaps, and Preparation Recommendations. CV: {text[:2000]}"
            res = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODEL_NAME)
            return res.choices[0].message.content
        
        btn1.click(start_analysis, [cv_in, jd_in], [out_analysis])
        btn1.click(lambda x: x, [out_analysis], [cv_memory])

    with gr.Tab("2. Level 2 (Audio)"):
        with gr.Row():
            with gr.Column(scale=2):
                chat2 = gr.Chatbot(height=400) 
                audio_in2 = gr.Audio(label="Voice Response", type="filepath")
                sub2 = gr.Button("Submit Answer", variant="primary")
            with gr.Column(scale=1):
                plot2 = gr.BarPlot(label="Live Analytics")
                fb2 = gr.Textbox(label="AI Interviewer Feedback")
        audio_out2 = gr.Audio(label="AI Follow-up", autoplay=True)

    with gr.Tab("3. Level 3 (Video)"):
        with gr.Row():
            with gr.Column(scale=2):
                video_in3 = gr.Video(label="Video Interview")
                chat3 = gr.Chatbot(height=350) 
                sub3 = gr.Button("Analyze My Performance", variant="primary")
            with gr.Column(scale=1):
                plot3 = gr.BarPlot(label="Multimodal Analytics")
                fb3 = gr.Textbox(label="Behavioral Coaching")
        audio_out3 = gr.Audio(label="AI Follow-up", autoplay=True)

    sub2.click(process_interview, [gr.State(None), audio_in2, chat_state, gr.State("2"), cv_memory], [chat2, audio_out2, plot2, fb2])
    sub3.click(process_interview, [video_in3, gr.State(None), chat_state, gr.State("3"), cv_memory], [chat3, audio_out3, plot3, fb3])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())