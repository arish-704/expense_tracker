import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from vosk import Model, KaldiRecognizer
import json
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import os
import tempfile
import wave
import av
import speech_recognition as sr
from dotenv import load_dotenv  # For API key management
import openai  # For GPT integration
from transformers import pipeline  # Fallback local AI




st.set_page_config(
    page_title="Smart Expense Tracker", 
    page_icon="💸", 
    layout="centered"
)





model = Model("vosk-model-small-en-us-0.15")
audio_queue = queue.Queue()
try:
    model = Model("vosk-model-small-en-us-0.15") if os.path.exists("vosk-model-small-en-us-0.15") else None
except Exception as e:
    st.error(f"Vosk initialization failed: {str(e)}")
    model = None





def init_db():
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    # Use triple-quoted string with standard SQL comments
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY,
            amount REAL,
            category TEXT,
            date TEXT DEFAULT CURRENT_TIMESTAMP,
            is_income INTEGER DEFAULT 0  -- 0=expense, 1=income (standard SQL comment)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            user_id INTEGER PRIMARY KEY,
            monthly_income REAL DEFAULT 30000
        )
    ''')
    conn.commit()
    conn.close()



# ---------- Function to Save Data ----------
def add_expense(amount, category):
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO expenses (amount, category, date) VALUES (?, ?, ?)",
        (amount, category, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()





def show_expenses():
    conn = sqlite3.connect('expenses.db')
    df = pd.read_sql("SELECT * FROM expenses", conn)
    st.dataframe(df)
    conn.close() 



# Audio frame handler



def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio = frame.to_ndarray()
    audio_queue.put(audio)
    return frame 



def save_audio_to_wav(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(48000)  # Streamlit WebRTC default
        wf.writeframes(audio_data.tobytes())

# Speech-to-text using Google STT
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Request Error from Google STT: {e}"

def voice_input():
    st.markdown("### 🎤 Speak now and wait for text to appear...")

    webrtc_ctx = webrtc_streamer(
        key="voice-input",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        on_audio_frame=audio_callback,
    )

    transcribed_text = None

    if webrtc_ctx.state.playing:
        st.success("Recording... speak clearly!")

        # Wait for a few seconds of audio
        if not audio_queue.empty():
            audio_frames = []
            for _ in range(80):  # Collect approx 2 seconds
                if not audio_queue.empty():
                    audio_frames.append(audio_queue.get())
            if audio_frames:
                all_audio = np.concatenate(audio_frames)

                # Save to a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    save_audio_to_wav(all_audio, tmpfile.name)
                    transcribed_text = transcribe_audio(tmpfile.name)

    return transcribed_text
    


def predict_spending():
    conn = sqlite3.connect('expenses.db')
    df = pd.read_sql("SELECT amount, date FROM expenses", conn)
    
    if len(df) < 5:
        return "Not enough data yet!"
    
    df['date'] = pd.to_datetime(df['date'])
    df['days'] = (df['date'] - df['date'].min()).dt.days
    
    model = LinearRegression()
    model.fit(df[['days']], df['amount'])
    
    future_day = df['days'].max() + 30  # Predict next month
    return model.predict([[future_day]])[0]  




def budget_tracker():
    conn = sqlite3.connect('expenses.db')
    
    # Get total expenses (negative) and income (positive)
    df = pd.read_sql("""
        SELECT 
            SUM(CASE WHEN is_income = 0 THEN amount ELSE 0 END) as total_expenses,
            SUM(CASE WHEN is_income = 1 THEN amount ELSE 0 END) as total_income
        FROM expenses
    """, conn)
    
    balance = df.iloc[0]["total_income"] - df.iloc[0]["total_expenses"]
    
    st.metric("Current Balance", f"₹{balance:,.2f}")
    
    budget = st.slider("Monthly Budget Goal", 1000, 100000, 30000)
    st.progress(min(balance / budget, 1.0))
    
    if balance < 0:
        st.error("You're overspending!")
    conn.close()
    





# ===== NEW FEATURE 1: INCOME TRACKING =====
def add_income(amount, source):
    """Track money received"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO expenses (amount, category, is_income) VALUES (?, ?, 1)",
        (amount, f"Income: {source}")
    )
    conn.commit()
    conn.close()



# ===== NEW FEATURE 2: DELETE ENTRIES =====
def delete_expense(expense_id):
    """Remove an entry"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    conn.close()



# ===== NEW FEATURE 3: GPT ADVISOR =====
def get_financial_advice(question):
    """Hybrid advisor with GPT-3.5 and local AI fallback"""
    try:
        load_dotenv()  # Load environment variables from .env
        openai.api_key = os.getenv("OPENAI_KEY")

        if not openai.api_key:
            raise ValueError("OpenAI API key not found in .env file.")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a financial advisor analyzing user expenses."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0]["message"]["content"]

    except Exception as e:
        print(f"[Warning] GPT API failed, using local fallback. Error: {e}")
        local_ai = pipeline("text-generation", model="gpt2")
        result = local_ai(question, max_length=100, do_sample=True)[0]["generated_text"]
        return result






def main():
     
    init_db()
    
     

    # ---------- CSS Styling ----------
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .title-style {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .stSelectbox, .stNumberInput {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown('<div class="title-style">💰 Smart Expense Tracker</div>', unsafe_allow_html=True)

    
    
    # ---------- Input Form ----------
    st.subheader("Add a New Expense")
    st.write("Fill in the details below to log your expense.")

    with st.form("expense_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
        with col2:
            category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Rent", "Other"])

        submit_button = st.form_submit_button(label="📂 Save Expense")

        if submit_button:
            add_expense(amount, category)
            st.success(f"✅ Logged ₹{amount} under *{category}*.")
    
    
    
    
    
    # ---------- Voice Input Option ----------
    st.markdown("### Or use voice to log your expense")
    st.info("🕒 After clicking, wait a moment before speaking. The mic takes a second to start.")
    st.title("🎙️ Real-time Voice Input to Text")

    result = voice_input()
    if result:
        st.markdown("### 📝 You said:")
        st.write(result)

    
    
     
     
     # ---------- Show Expense History ----------
    st.subheader("📊 Expense History")
    show_expenses()

    
    
    # ---------- Budget Tracker ----------
    st.subheader("📉 Monthly Budget Tracker")
    budget_tracker()

    
    
    # ---------- Spending Prediction ----------
    st.subheader("🔮 Predict Future Spending")
    if st.button("🔮 Predict Next Month's Spending"):
        prediction = predict_spending()
        if isinstance(prediction, str):
            st.warning(prediction)
        else:
            st.success(f"📈 Predicted spending for next 30 days: ₹{prediction:.2f}")
    

    # ===== INCOME TRACKING UI =====
with st.expander("💵 Add Income (Money Received)"):
    income_amount = st.number_input("Amount Received", min_value=0.0, key="income_amt")
    income_source = st.text_input("Source (e.g., 'Client Payment', 'Loan Return')")
    if st.button("Add Income"):
        add_income(income_amount, income_source)
        st.success(f"➕ Added ₹{income_amount} from {income_source}")

# ===== EXPENSE DELETION UI =====
st.subheader("🗑️ Manage Entries")
with sqlite3.connect('expenses.db') as conn:  # <-- Auto-closes connection
    df = pd.read_sql("SELECT id, amount, category, date FROM expenses", conn)
    edited_df = st.data_editor(df)

    if st.button("Delete Selected"):
        for index, row in edited_df.iterrows():
            delete_expense(row['id'])
        st.rerun()

# ===== GPT ADVISOR UI =====
st.subheader("🤖 AI Financial Advisor")
advisor_question = st.text_input("Ask anything (e.g., 'Can I afford a ₹25k phone?')")
if advisor_question:
    advice = get_financial_advice(advisor_question)
    st.markdown(f"**💡 Advice:** {advice}")


    
    
    
    # ---------- Footer ----------
    st.markdown("---")
    st.caption("Designed with ❤️ using Streamlit | Track smart, spend smarter!")






if __name__ == "__main__":
    main()
