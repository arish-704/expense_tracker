import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
from dotenv import load_dotenv
import openai
from transformers import pipeline
import os
from openai import OpenAIError, APIConnectionError, AuthenticationError
from openai import OpenAI
# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💸",
    layout="centered"
)

# ===== DATABASE FUNCTIONS =====
def init_db():
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    # Force new schema
    cursor.execute("DROP TABLE IF EXISTS expenses")
    
    cursor.execute('''
        CREATE TABLE expenses (
            id INTEGER PRIMARY KEY,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            date TEXT DEFAULT CURRENT_TIMESTAMP,
            is_income INTEGER DEFAULT 0  -- 0=expense, 1=income
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
    print("New database schema created")

def add_expense(amount, category):
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, is_income) VALUES (?, ?, 0)",
                (amount, category)
            )
            conn.commit()  # Explicit commit
            print(f"Added expense: ₹{amount} for {category}")
    except Exception as e:
        print(f"Error adding expense: {e}")
        raise  # Re-raise to show error in UI

def add_income(amount, source):
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, is_income) VALUES (?, ?, 1)",
                (amount, f"Income: {source}")
            )
            conn.commit()  # Explicit commit
            print(f"Added income: ₹{amount} from {source}")
    except Exception as e:
        print(f"Error adding income: {e}")
        raise


def delete_expense(expense_id):
    with sqlite3.connect('expenses.db') as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))

# ===== FINANCIAL FUNCTIONS =====
def predict_spending():
    with sqlite3.connect('expenses.db') as conn:
        df = pd.read_sql("SELECT amount, date FROM expenses WHERE is_income = 0", conn)
        
        if len(df) < 5:
            return "Not enough data yet!"
        
        df['date'] = pd.to_datetime(df['date'])
        df['days'] = (df['date'] - df['date'].min()).dt.days
        
        model = LinearRegression()
        model.fit(df[['days']], df['amount'])
        
        future_day = df['days'].max() + 30
        return model.predict([[future_day]])[0]

def budget_tracker():
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql("""
                SELECT 
                    SUM(CASE WHEN is_income = 0 THEN amount ELSE 0 END) as expenses,
                    SUM(CASE WHEN is_income = 1 THEN amount ELSE 0 END) as income
                FROM expenses
            """, conn)
            
            expenses = df.iloc[0]["expenses"]
            income = df.iloc[0]["income"]

            # Handle None values (i.e., no income or expense yet)
            if expenses is None:
                expenses = 0.0
            if income is None:
                income = 0.0

            balance = income - expenses
            
            # Display current balance
            st.metric("💼 Current Balance", f"₹{balance:,.2f}")

            # Monthly budget input
            budget = st.slider("📊 Monthly Budget Goal", 1000, 100000, 30000)

            # Progress bar calculation (clamped between 0 and 1)
            progress_value = min(max(balance / budget, 0.0), 1.0)
            st.progress(progress_value)

            # Overspending warning
            if balance < 0:
                st.error("🚨 You're overspending!")

    except Exception as e:
        st.error(f"⚠️ Error calculating budget: {str(e)}")


# ===== VOICE INPUT =====
def voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Speak now... (say something like '500 rupees for food')")

            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)

            if "for" in text.lower():
                parts = text.lower().split("for")
                amount = float(parts[0].replace("rs", "").replace("rupees", "").strip())
                category = parts[1].strip().title()

                st.session_state["voice_amount"] = amount
                st.session_state["voice_category"] = category
                st.success(f"Recognized: ₹{amount} for {category}")
                st.rerun()
            else:
                st.warning("Please say amount followed by 'for <category>'")
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except Exception as e:
        st.error(f"Error: {str(e)}")



# ===== AI ADVISOR =====
@st.cache_resource
def load_advisor_model():
    return pipeline("text-generation", model="gpt2")

def get_financial_advice(question):
    try:
        generator = load_advisor_model()
        prompt = f"As a smart financial advisor, suggest advice for: {question}"
        response = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True)
        return response[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Error using local advisor model: {e}"
    

# ===== MAIN APP =====
def main():
    # Enhanced Custom CSS
    st.markdown("""
        <style>
        /* Global Styles */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f0f4f8;
        }
        .main {
            background: linear-gradient(to bottom right, #f0f4f8, #e0ecf7);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .title-style {
            font-size: 45px;
            font-weight: 800;
            text-align: center;
            color: #2c3e50;
            padding: 20px 10px;
            margin-bottom: 30px;
            background: -webkit-linear-gradient(#2c3e50, #3498db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        /* Custom form sections */
        .stForm {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        /* Buttons */
        button[kind="primary"] {
            background-color: #3498db !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.2rem !important;
            font-weight: 600 !important;
        }
        button[kind="primary"]:hover {
            background-color: #2c80b4 !important;
            transform: scale(1.03);
            transition: 0.3s;
        }
        /* Tabs styling */
        .stTabs [role="tablist"] {
            background-color: #dfeeff;
            border-radius: 10px;
            padding: 5px;
            margin-bottom: 10px;
        }
        .stTabs [role="tab"] {
            font-weight: 600;
            color: #2c3e50;
        }
        /* DataFrame styling */
        .stDataFrameContainer {
            border-radius: 12px !important;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-style">💰 Smart Expense Tracker</div>', unsafe_allow_html=True)

    # === TRANSACTION INPUT ===
    st.subheader("Add Transaction")
    tab1, tab2 = st.tabs(["💰 Expense", "💵 Income"])
    
    with tab1:
        with st.form("expense_form"):
            amount = st.number_input(
                "Amount (₹)", 
                min_value=0.0, 
                step=100.0, 
                key="expense_amt",
                value=st.session_state.get("voice_amount", 0.0)
            )
            category = st.selectbox(
                "Category", 
                ["Food", "Transport", "Entertainment", "Rent", "Other"],
                index=["Food", "Transport", "Entertainment", "Rent", "Other"].index(
                    st.session_state.get("voice_category", "Food")
                )
            )
            if st.form_submit_button("Add Expense"):
                add_expense(amount, category)
                st.success(f"Added ₹{amount} expense")
                if "voice_amount" in st.session_state:
                    del st.session_state.voice_amount
                if "voice_category" in st.session_state:
                    del st.session_state.voice_category
        
    with tab2:
        with st.form("income_form"):
            income_amt = st.number_input("Amount Received", min_value=0.0, key="income_amt")
            source = st.text_input("Source")
            if st.form_submit_button("Add Income"):
                add_income(income_amt, source)
                st.success(f"Added ₹{income_amt} income")

    # === VOICE INPUT ===
    if st.button("🎤 Start Voice Input"):
        voice_input()
    
    # === TRANSACTION MANAGEMENT ===
    with sqlite3.connect('expenses.db') as conn:
        df = pd.read_sql("SELECT id, amount, category, date FROM expenses ORDER BY date DESC", conn)
        if len(df) > 0:
            df["Delete"] = False
            edited_df = st.data_editor(
                df,
                column_config={
                    "Delete": st.column_config.CheckboxColumn(required=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("Delete Selected", type="primary"):
                ids_to_delete = edited_df[edited_df["Delete"]]["id"].tolist()
                for expense_id in ids_to_delete:
                    delete_expense(expense_id)
                st.rerun()
        else:
            st.warning("No transactions yet. Add one using the form or voice input!")

    # === DASHBOARD ===
    st.subheader("Financial Overview")
    budget_tracker()
    
    if st.button("🔮 Predict Next Month's Spending"):
        prediction = predict_spending()
        if isinstance(prediction, str):
            st.warning(prediction)
        else:
            st.success(f"Predicted spending: ₹{prediction:.2f}")

    # === AI ADVISOR ===
    st.subheader("AI Financial Advisor")
    question = st.text_input("Ask a financial question")
    if question:
        with st.spinner("Analyzing..."):
            advice = get_financial_advice(question)
            st.write(advice)


if __name__ == "__main__":
    main()
