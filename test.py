import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
from transformers import pipeline
import json
import os
from google import genai
from transformers import pipeline
from sklearn.ensemble import IsolationForest

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💸",
    layout="wide"
)

# ===== DATABASE FUNCTIONS =====
def init_db():
    """Initialize database with proper schema"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()

    # Expenses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            date TEXT DEFAULT CURRENT_TIMESTAMP,
            is_income INTEGER DEFAULT 0
        )
    ''')

    # Settings table (ready for future persistence)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            user_id INTEGER PRIMARY KEY,
            monthly_income REAL DEFAULT 30000
        )
    ''')

    # Split expenses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS split_expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL,
            description TEXT NOT NULL,
            total_amount REAL NOT NULL,
            paid_by TEXT NOT NULL,
            participants TEXT NOT NULL,
            date TEXT DEFAULT CURRENT_TIMESTAMP,
            settled INTEGER DEFAULT 0
        )
    ''')

    # Financial goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_name TEXT NOT NULL,
            target_amount REAL NOT NULL,
            current_amount REAL DEFAULT 0,
            deadline TEXT,
            category TEXT DEFAULT 'General',
            created_date TEXT DEFAULT CURRENT_TIMESTAMP,
            completed INTEGER DEFAULT 0
        )
    ''')

    # Achievements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badge_name TEXT NOT NULL,
            description TEXT,
            earned_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Budget alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS budget_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            budget_limit REAL NOT NULL,
            alert_threshold INTEGER DEFAULT 80
        )
    ''')

    # Ensure one alert per category (supports UPSERT)
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS ux_budget_alerts_category
        ON budget_alerts(category)
    ''')

    # Streaks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saving_streaks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            saved INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()

def add_expense(amount, category):
    """Add an expense to the database"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, is_income) VALUES (?, ?, 0)",
                (amount, category)
            )
            conn.commit()
            check_budget_alerts(category, amount)
            return True
    except Exception as e:
        st.error(f"Error adding expense: {e}")
        return False

def add_income(amount, source):
    """Add income to the database"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, is_income) VALUES (?, ?, 1)",
                (amount, f"Income: {source}")
            )
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error adding income: {e}")
        return False

def delete_expense(expense_id):
    """Delete an expense by ID"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error deleting expense: {e}")
        return False

# ===== SPLIT EXPENSES FUNCTIONS =====
def add_split_expense(group_name, description, total_amount, paid_by, participants):
    """Add a split expense"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            participants_json = json.dumps(participants)
            cursor.execute(
                """INSERT INTO split_expenses
                (group_name, description, total_amount, paid_by, participants)
                VALUES (?, ?, ?, ?, ?)""",
                (group_name, description, total_amount, paid_by, participants_json)
            )
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error adding split expense: {e}")
        return False

def get_split_expenses(settled=False):
    """Get all split expenses"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql(
                "SELECT * FROM split_expenses WHERE settled = ? ORDER BY date DESC",
                conn,
                params=(1 if settled else 0,)
            )
            return df
    except Exception as e:
        st.error(f"Error getting split expenses: {e}")
        return pd.DataFrame()

def settle_split_expense(expense_id):
    """Mark a split expense as settled"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE split_expenses SET settled = 1 WHERE id = ?", (expense_id,))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error settling expense: {e}")
        return False

def calculate_split_balances(group_name=None):
    """Calculate who owes whom"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            base = "SELECT * FROM split_expenses WHERE settled = 0"
            if group_name:
                df = pd.read_sql(base + " AND group_name = ?", conn, params=(group_name,))
            else:
                df = pd.read_sql(base, conn)

            if len(df) == 0:
                return {}

            balances = {}
            for _, row in df.iterrows():
                paid_by = row['paid_by']
                total = row['total_amount']
                participants = json.loads(row['participants'])

                share = total / max(1, len(participants))

                balances[paid_by] = balances.get(paid_by, 0.0) + total
                for person in participants:
                    balances[person] = balances.get(person, 0.0) - share

            return balances
    except Exception as e:
        st.error(f"Error calculating balances: {e}")
        return {}

# ===== FINANCIAL GOALS FUNCTIONS =====
def add_goal(goal_name, target_amount, deadline, category):
    """Add a financial goal"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO financial_goals
                (goal_name, target_amount, deadline, category)
                VALUES (?, ?, ?, ?)""",
                (goal_name, target_amount, deadline, category)
            )
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error adding goal: {e}")
        return False

def update_goal_progress(goal_id, amount_to_add):
    """Update goal progress"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE financial_goals SET current_amount = current_amount + ? WHERE id = ?",
                (amount_to_add, goal_id)
            )

            cursor.execute(
                "SELECT current_amount, target_amount, goal_name FROM financial_goals WHERE id = ?",
                (goal_id,)
            )
            result = cursor.fetchone()

            if result and result[0] >= result[1]:
                cursor.execute("UPDATE financial_goals SET completed = 1 WHERE id = ?", (goal_id,))
                add_achievement(f"Goal Achieved: {result[2]}", f"Completed your {result[2]} goal!")

            conn.commit()
            update_streak()
            return True
    except Exception as e:
        st.error(f"Error updating goal: {e}")
        return False

def get_goals():
    """Get all financial goals"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql(
                "SELECT * FROM financial_goals WHERE completed = 0 ORDER BY deadline",
                conn
            )
            return df
    except Exception:
        return pd.DataFrame()

def delete_goal(goal_id):
    """Delete a goal"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM financial_goals WHERE id = ?", (goal_id,))
            conn.commit()
            return True
    except Exception:
        return False

# ===== GAMIFICATION FUNCTIONS =====
def add_achievement(badge_name, description):
    """Add an achievement badge"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM achievements WHERE badge_name = ?", (badge_name,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO achievements (badge_name, description) VALUES (?, ?)",
                    (badge_name, description)
                )
                conn.commit()
                return True
    except Exception:
        return False
    return False

def get_achievements():
    """Get all achievements"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql("SELECT * FROM achievements ORDER BY earned_date DESC", conn)
            return df
    except Exception:
        return pd.DataFrame()

def update_streak():
    """Update saving streak with correct weekly and monthly badges"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO saving_streaks (date, saved) VALUES (?, 1)",
                (today,)
            )
            conn.commit()

            # 7-day window for weekly badge
            cursor.execute(
                "SELECT COUNT(*) FROM saving_streaks WHERE date >= date('now', '-6 days')"
            )
            last7 = cursor.fetchone()[0] or 0

            if last7 >= 7:
                add_achievement("Week Warrior", "Saved for 7 consecutive days!")

            # 30-day window for monthly badge
            cursor.execute(
                "SELECT COUNT(*) FROM saving_streaks WHERE date >= date('now', '-29 days')"
            )
            last30 = cursor.fetchone()[0] or 0

            if last30 >= 30:
                add_achievement("Month Master", "Saved for 30 consecutive days!")
    except Exception:
        pass

def get_current_streak():
    """Get current saving streak (last 30 days active days)"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM saving_streaks
                WHERE date >= date('now', '-29 days')
            """)
            result = cursor.fetchone()
            return result[0] if result else 0
    except Exception:
        return 0

# ===== BUDGET ALERTS FUNCTIONS =====
def add_budget_alert(category, budget_limit, threshold=80):
    """Add a budget alert for a category with UPSERT"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO budget_alerts(category, budget_limit, alert_threshold)
                VALUES (?, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    budget_limit = excluded.budget_limit,
                    alert_threshold = excluded.alert_threshold
            """, (category, budget_limit, threshold))
            conn.commit()
            return True
    except Exception:
        return False

def check_budget_alerts(category, new_expense_amount):
    """Check if any budget alerts should trigger"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT budget_limit, alert_threshold FROM budget_alerts WHERE category = ?",
                (category,)
            )
            result = cursor.fetchone()

            if result:
                budget_limit, threshold = result

                cursor.execute("""
                    SELECT SUM(amount) FROM expenses
                    WHERE category = ? AND is_income = 0
                    AND date >= date('now', 'start of month')
                """, (category,))
                total_spent = cursor.fetchone()[0] or 0.0

                percentage = (total_spent / budget_limit) * 100 if budget_limit > 0 else 0

                if percentage >= threshold:
                    st.warning(f"⚠️ Alert: You've spent {percentage:.1f}% of your {category} budget (₹{total_spent:,.2f} / ₹{budget_limit:,.2f})")
                if percentage >= 100:
                    st.error(f"🚨 Budget Exceeded: {category} budget exceeded by ₹{total_spent - budget_limit:,.2f}!")
    except Exception:
        pass

def get_budget_alerts():
    """Get all budget alerts"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql("SELECT * FROM budget_alerts", conn)
            return df
    except Exception:
        return pd.DataFrame()

# ===== FINANCIAL FUNCTIONS =====
def predict_spending():
    """Predict next month's total spending with robust fallbacks"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql("SELECT amount, date FROM expenses WHERE is_income = 0", conn)
        if len(df) < 1:
            return None, "No expense data yet. Add some expenses to start forecasting."
        df['date'] = pd.to_datetime(df['date'])

        # Monthly totals
        monthly = df.groupby(pd.Grouper(key='date', freq='MS'))['amount'].sum().reset_index()
        m = len(monthly)

        # 3+ months: linear regression on month index (reduces sensitivity to day-length)
        if m >= 3:
            monthly['t'] = np.arange(m)
            model = LinearRegression().fit(monthly[['t']], monthly['amount'])
            yhat = float(model.predict([[m]])[0])
            last = float(monthly['amount'].iloc[-1])
            return max(0.0, yhat), None  # clamp negatives to 0

        # 2 months: drift baseline (line through first and last)
        if m == 2:
            y1, yT = float(monthly['amount'].iloc[0]), float(monthly['amount'].iloc[1])
            drift = yT + (yT - y1)  # next = last + average change
            return max(0.0, drift), None

        # 1 month: naïve baseline (next = last month)
        if m == 1:
            return float(monthly['amount'].iloc[0]), None

        return None, "Not enough data to forecast."
    except Exception as e:
        return None, f"Error predicting spending: {str(e)}"


def budget_tracker():
    """Display budget tracking dashboard (monthly-aware with inline prediction)"""
    try:
        # Use absolute DB path if defined elsewhere, otherwise local file
        try:
            DB = DB_PATH  # defined in your app if you adopted the absolute path fix
        except NameError:
            DB = 'expenses.db'

        with sqlite3.connect(DB) as conn:
            # All-time totals (for balance)
            df_totals = pd.read_sql("""
                SELECT
                    SUM(CASE WHEN is_income = 1 THEN amount ELSE 0 END) AS income,
                    SUM(CASE WHEN is_income = 0 THEN amount ELSE 0 END) AS expenses
                FROM expenses
            """, conn)

            # Current-month totals (for budget progress)
            df_month = pd.read_sql("""
                SELECT
                    SUM(CASE WHEN is_income = 1 THEN amount ELSE 0 END) AS m_income,
                    SUM(CASE WHEN is_income = 0 THEN amount ELSE 0 END) AS m_expenses
                FROM expenses
                WHERE date >= date('now','start of month')
            """, conn)

        # All-time metrics
        expenses_all = float(df_totals.iloc[0]["expenses"] or 0.0)
        income_all   = float(df_totals.iloc[0]["income"]   or 0.0)
        balance_all  = income_all - expenses_all

        # Monthly metrics for progress vs monthly budget
        month_expenses = float(df_month.iloc[0]["m_expenses"] or 0.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Total Income", f"₹{income_all:,.2f}")
        with col2:
            st.metric("💸 Total Expenses", f"₹{expenses_all:,.2f}")
        with col3:
            st.metric("💼 Balance", f"₹{balance_all:,.2f}")

        # Keep your existing slider; now progress uses current-month spending
        budget = st.slider("📊 Monthly Budget Goal", 1000, 100000, 30000, step=1000)

        if budget > 0:
            progress_value = min(max(month_expenses / budget, 0.0), 1.0)
            st.progress(progress_value)
            st.caption(f"You've spent {(month_expenses / budget * 100):.1f}% of your budget")

        # Warnings (all-time overspend still shown; budget overspend uses monthly)
        if balance_all < 0:
            st.error("🚨 You're overspending! Your expenses exceed your income.")
        elif month_expenses > budget:
            st.warning(f"⚠️ You've exceeded your budget by ₹{month_expenses - budget:,.2f}")

        # Inline prediction so this section can show forecasts without relying on elsewhere
        if st.button("🔮 Predict Next Month's Spending", use_container_width=True, key="btn_predict_overview"):
            try:
                prediction, error = predict_spending()
                if error:
                    st.warning(error)
                else:
                    st.success(f"📊 Predicted spending for next month: ₹{prediction:,.2f}")
            except Exception as e:
                st.error(f"⚠️ Error predicting spending: {e}")

    except Exception as e:
        st.error(f"⚠️ Error calculating budget: {str(e)}")



# ===== VOICE INPUT =====
def voice_input():
    """Process voice input and auto-save expense using BERT classification"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Say something like '400 at Dominos' or '50 for chai'")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        st.info("Processing speech...")
        text = recognizer.recognize_google(audio)
        st.write(f"You said: '{text}'")
        
        # Extract amount using regex
        import re
        nums = re.findall(r'\d+\.?\d*', text)
        if not nums:
            st.warning("⚠️ Could not detect an amount. Please try again.")
            return False
        amount = float(nums[0])
        
        # Remove amount from text to get description
        description = re.sub(r'\d+\.?\d*', '', text).strip()
        # Remove common filler words
        description = re.sub(r'\b(at|for|to|rupees?|rs?)\b', '', description, flags=re.IGNORECASE).strip()
        
        if not description:
            st.warning("⚠️ Could not understand the expense description. Please try again.")
            return False
        
        # Use BERT to classify the description into category
        categories = ["Food", "Transport", "Entertainment", "Rent", "Utilities", "Shopping", "Healthcare", "Other"]
        try:
            sug_label, sug_score = suggest_category_with_bert(description, categories)
            st.success(f"✅ Detected: ₹{amount} → {sug_label} (confidence {sug_score:.2f})")
            
            # Auto-save the expense directly
            if add_expense(amount, sug_label):
                st.success(f"💾 Saved ₹{amount} as {sug_label} expense!")
                return True
            else:
                st.error("Failed to save expense")
                return False
        except Exception as e:
            st.error(f"BERT classification failed: {e}")
            return False
            
    except sr.WaitTimeoutError:
        st.error("⏱️ Listening timed out. Please try again.")
        return False
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio. Please speak clearly.")
        return False
    except sr.RequestError:
        st.error("🌐 Network error with speech service. Check your internet and try again.")
        return False
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


# ===== AI ADVISOR =====
@st.cache_resource
def get_gemini_client():
    # Read from Streamlit secrets first, then from environment
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml or your environment.")
        return None
    return genai.Client(api_key=api_key)

def get_financial_advice(question: str) -> str:
    client = get_gemini_client()
    if client is None:
        return "AI advisor is unavailable until a Gemini API key is configured."
    try:
        # Fast default model; you can switch to 'gemini-2.0-pro' for deeper answers
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=question.strip()
        )
        text = getattr(resp, "text", None)
        return text.strip() if text else "No response generated. Try rephrasing your question."
    except Exception as e:
        return f"Error from Gemini API: {e}"
    

    # ===== BERT ======

@st.cache_resource
def load_bert_zero_shot():
    """Load zero-shot classification pipeline with a stable model"""
    # facebook/bart-large-mnli is the standard, reliable model for zero-shot classification
    # Alternative: "typeform/distilbert-base-uncased-mnli" (smaller, faster but less accurate)
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )


def suggest_category_with_bert(text: str, candidate_labels: list[str]) -> tuple[str, float]:
    clf = load_bert_zero_shot()
    out = clf(text, candidate_labels=candidate_labels, multi_label=False)
    return out["labels"][0], float(out["scores"][0])


# ====== isolation forest ======
@st.cache_resource
def load_anomaly_detector():
    """Train Isolation Forest on historical expense amounts"""
    try:
        with sqlite3.connect('expenses.db') as conn:
            df = pd.read_sql("SELECT amount FROM expenses WHERE is_income = 0", conn)
        
        if len(df) < 10:
            return None  # Need at least 10 expenses to train
        
        amounts = df['amount'].values.reshape(-1, 1)
        
        # Increased contamination to 0.1 (10% outlier rate) for better sensitivity
        # Lower contamination = fewer outliers flagged, higher = more sensitive
        model = IsolationForest(
            contamination=0.1,  # Expect 10% of data to be outliers (was 0.05)
            random_state=42,
            n_estimators=100  # More trees = more stable predictions
        )
        model.fit(amounts)
        return model
    except Exception as e:
        st.error(f"Error training anomaly detector: {e}")
        return None

def check_expense_anomaly(amount: float) -> tuple[bool, str]:
    """Check if an expense amount is anomalous"""
    detector = load_anomaly_detector()
    if detector is None:
        return False, ""
    
    try:
        prediction = detector.predict([[amount]])[0]
        score = detector.score_samples([[amount]])[0]
        
        if prediction == -1:  # Anomaly detected
            return True, f"⚠️ Unusual amount detected! This is {abs(score):.1f}× more unusual than normal expenses. Please verify."
        return False, ""
    except Exception as e:
        st.error(f"Error checking anomaly: {e}")
        return False, ""


# ===== MAIN APP =====
def main():
    # Initialize database
    init_db()

    # Enhanced Custom CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to bottom right, #f0f4f8, #e0ecf7);
            padding: 20px;
        }
        .title-style {
            font-size: 45px;
            font-weight: 800;
            text-align: center;
            color: #2c3e50;
            padding: 20px 10px;
            margin-bottom: 30px;
            background: linear-gradient(90deg, #2c3e50, #3498db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stForm {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .achievement-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            margin: 5px;
            display: inline-block;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-style">💰 Smart Expense Tracker</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Dashboard", "💸 Transactions", "🤝 Split Expenses", "🎯 Financial Goals", "🔔 Alerts & Notifications", "🤖 AI Advisor"]
    )

    # Show current streak in sidebar
    streak = get_current_streak()
    st.sidebar.metric("🔥 Saving Streak", f"{streak} days")

    # Show achievements in sidebar
    achievements_df = get_achievements()
    if len(achievements_df) > 0:
        st.sidebar.subheader("🏆 Recent Achievements")
        for _, achievement in achievements_df.head(3).iterrows():
            st.sidebar.markdown(f"<div class='achievement-badge'>🏅 {achievement['badge_name']}</div>", unsafe_allow_html=True)

    # ===== DASHBOARD PAGE =====
    if page == "🏠 Dashboard":
        st.header("📊 Financial Overview")
        budget_tracker()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔮 Spending Prediction")
            if st.button("Predict Next Month", use_container_width=True):
                prediction, error = predict_spending()
                if error:
                    st.warning(error)
                else:
                    st.success(f"📊 Predicted: ₹{prediction:,.2f}")

        with col2:
            st.subheader("🎯 Active Goals")
            goals_df = get_goals()
            if len(goals_df) > 0:
                st.metric("Goals in Progress", len(goals_df))
            else:
                st.info("No active goals. Create one in the Goals section!")

    # ===== TRANSACTIONS PAGE =====
    elif page == "💸 Transactions":
        st.header("📝 Manage Transactions")

        tab1, tab2 = st.tabs(["💸 Add Transaction", "📊 View History"])

        with tab1:
            subtab1, subtab2 = st.tabs(["Expense", "Income"])

            with subtab1:
                with st.form("expense_form", clear_on_submit=True):
                    categories = ["Food", "Transport", "Entertainment", "Rent", "Utilities", "Shopping", "Healthcare", "Other"]

                    voice_amount = st.session_state.get("voice_amount", 0.0)
                    voice_category = st.session_state.get("voice_category", "Food")

                    try:
                        category_index = categories.index(voice_category) if voice_category in categories else 0
                    except ValueError:
                        category_index = 0

                    amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0, value=voice_amount)
                    category = st.selectbox("Category", categories, index=category_index)

                    submitted = st.form_submit_button("Add Expense", type="primary", use_container_width=True)

                    if submitted and amount > 0:
                        # NEW: Check for anomaly before saving
                        is_anomaly, msg = check_expense_anomaly(amount)
                        if is_anomaly:
                            st.warning(msg)
                        
                        if add_expense(amount, category):
                            st.success(f"✅ Added ₹{amount:,.2f} expense for {category}")
                            if "voice_amount" in st.session_state:
                                del st.session_state.voice_amount
                            if "voice_category" in st.session_state:
                                del st.session_state.voice_category
                            st.rerun()

            with subtab2:
                with st.form("income_form", clear_on_submit=True):
                    income_amt = st.number_input("Amount Received (₹)", min_value=0.0, step=100.0)
                    source = st.text_input("Source (e.g., Salary, Freelance)")

                    submitted = st.form_submit_button("Add Income", type="primary", use_container_width=True)

                    if submitted and income_amt > 0 and source:
                        if add_income(income_amt, source):
                            st.success(f"✅ Added ₹{income_amt:,.2f} income from {source}")
                            st.rerun()
            
            

            st.divider()
            if st.button("🎤 Add by Voice", use_container_width=True):
                if voice_input():
                    st.rerun()
            
            st.divider()
            if st.button("🧪 Test Anomaly Detector (Debug)", use_container_width=True):
                detector = load_anomaly_detector()
                if detector is None:
                    st.error("Detector not trained yet. Add at least 10 expenses first.")
                else:
                    st.success("✅ Detector is trained!")
                    
                    # Test with a huge outlier
                    test_amounts = [100, 500, 1000, 300000]  # 3 lakh should be flagged
                    st.write("Testing amounts:")
                    for amt in test_amounts:
                        is_anom, msg = check_expense_anomaly(amt)
                        if is_anom:
                            st.error(f"₹{amt:,.0f}: {msg}")
                        else:
                            st.info(f"₹{amt:,.0f}: Normal expense")

        with tab2:
            try:
                with sqlite3.connect('expenses.db') as conn:
                    df = pd.read_sql("""
                        SELECT
                            id, amount, category, date,
                            CASE WHEN is_income = 1 THEN 'Income' ELSE 'Expense' END as type
                        FROM expenses ORDER BY date DESC
                    """, conn)

                    if len(df) > 0:
                        df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")
                        df["Delete"] = False

                        edited_df = st.data_editor(
                            df,
                            column_config={"Delete": st.column_config.CheckboxColumn(required=True)},
                            hide_index=True,
                            use_container_width=True,
                            disabled=["id", "amount", "category", "date", "type"]
                        )

                        if st.button("🗑️ Delete Selected"):
                            ids_to_delete = edited_df[edited_df["Delete"]]["id"].tolist()
                            if ids_to_delete:
                                for expense_id in ids_to_delete:
                                    delete_expense(expense_id)
                                st.success(f"Deleted {len(ids_to_delete)} transaction(s)")
                                st.rerun()
                            else:
                                st.warning("No transactions selected for deletion")
                    else:
                        st.info("📭 No transactions yet!")
            except Exception as e:
                st.error(f"Error: {e}")





    # ===== SPLIT EXPENSES PAGE =====
    elif page == "🤝 Split Expenses":
        st.header("🤝 Split Expenses with Friends")

        tab1, tab2, tab3 = st.tabs(["➕ Add Split", "📊 View Splits", "💰 Balances"])

        with tab1:
            with st.form("split_expense_form", clear_on_submit=True):
                st.subheader("Create a Split Expense")

                group_name = st.text_input("Group/Trip Name", placeholder="e.g., Goa Trip, Roommates")
                description = st.text_input("Description", placeholder="e.g., Dinner at restaurant")
                total_amount = st.number_input("Total Amount (₹)", min_value=0.0, step=100.0)

                st.write("**Participants:**")
                num_participants = st.number_input("Number of people", min_value=2, max_value=10, value=2)

                participants = []
                paid_by = None

                cols = st.columns(2)
                for i in range(num_participants):
                    with cols[i % 2]:
                        name = st.text_input(f"Person {i+1}", key=f"participant_{i}")
                        if name:
                            participants.append(name)

                if participants:
                    paid_by = st.selectbox("Who paid?", participants)

                submitted = st.form_submit_button("Add Split Expense", type="primary")

                if submitted and group_name and description and total_amount > 0 and len(participants) >= 2 and paid_by:
                    if add_split_expense(group_name, description, total_amount, paid_by, participants):
                        st.success(f"✅ Added split expense: {description}")
                        st.rerun()
                elif submitted:
                    st.error("Please fill all fields correctly!")

        with tab2:
            st.subheader("📋 All Split Expenses")

            splits_df = get_split_expenses(settled=False)

            if len(splits_df) > 0:
                for _, split in splits_df.iterrows():
                    with st.expander(f"🎫 {split['description']} - ₹{split['total_amount']:,.2f}"):
                        st.write(f"**Group:** {split['group_name']}")
                        st.write(f"**Paid by:** {split['paid_by']}")
                        st.write(f"**Date:** {split['date']}")

                        participants = json.loads(split['participants'])
                        share_per_person = split['total_amount'] / max(1, len(participants))

                        st.write(f"**Share per person:** ₹{share_per_person:,.2f}")
                        st.write(f"**Participants:** {', '.join(participants)}")

                        if st.button(f"Mark as Settled", key=f"settle_{split['id']}"):
                            if settle_split_expense(split['id']):
                                st.success("Marked as settled!")
                                st.rerun()
            else:
                st.info("No active split expenses. Add one above!")

            with st.expander("📦 View Settled Expenses"):
                settled_df = get_split_expenses(settled=True)
                if len(settled_df) > 0:
                    st.dataframe(settled_df[['group_name', 'description', 'total_amount', 'date']], use_container_width=True)
                else:
                    st.info("No settled expenses yet.")

        with tab3:
            st.subheader("💰 Who Owes Whom")

            splits_df = get_split_expenses(settled=False)
            if len(splits_df) > 0:
                groups = splits_df['group_name'].unique().tolist()

                selected_group = st.selectbox("Select Group", ["All Groups"] + groups)

                group_filter = None if selected_group == "All Groups" else selected_group
                balances = calculate_split_balances(group_filter)

                if balances:
                    st.write("**Current Balances:**")

                    owes = {k: v for k, v in balances.items() if v < 0}
                    gets_paid = {k: v for k, v in balances.items() if v > 0}

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**🔴 Owes Money:**")
                        for person, amount in sorted(owes.items(), key=lambda x: x[1]):
                            st.error(f"{person}: ₹{abs(amount):,.2f}")

                    with col2:
                        st.write("**🟢 Gets Paid:**")
                        for person, amount in sorted(gets_paid.items(), key=lambda x: x[1], reverse=True):
                            st.success(f"{person}: ₹{amount:,.2f}")

                    st.divider()
                    st.write("**💡 Settlement Suggestions:**")

                    # Greedy settlement algorithm with in-place updates
                    owes_list = [[k, abs(v)] for k, v in owes.items()]
                    gets_list = [[k, v] for k, v in gets_paid.items()]

                    i, j = 0, 0
                    while i < len(owes_list) and j < len(gets_list):
                        ower, owe_amt = owes_list[i]
                        getter, get_amt = gets_list[j]

                        payment = min(owe_amt, get_amt)
                        if payment > 0:
                            st.info(f"💸 {ower} pays ₹{payment:,.2f} to {getter}")
                            owes_list[i][1] -= payment
                            gets_list[j][1] -= payment

                        if owes_list[i][1] <= 1e-6:
                            i += 1
                        if gets_list[j][1] <= 1e-6:
                            j += 1
                else:
                    st.info("All settled! No pending balances.")
            else:
                st.info("No active split expenses to calculate balances.")

    # ===== FINANCIAL GOALS PAGE =====
    elif page == "🎯 Financial Goals":
        st.header("🎯 Financial Goals & Gamification")

        tab1, tab2, tab3 = st.tabs(["➕ Create Goal", "📊 My Goals", "🏆 Achievements"])

        with tab1:
            with st.form("goal_form", clear_on_submit=True):
                st.subheader("Set a New Financial Goal")

                col1, col2 = st.columns(2)

                with col1:
                    goal_name = st.text_input("Goal Name", placeholder="e.g., Vacation Fund, New Laptop")
                    target_amount = st.number_input("Target Amount (₹)", min_value=100.0, step=500.0)

                with col2:
                    goal_category = st.selectbox(
                        "Category",
                        ["Vacation", "Electronics", "Emergency Fund", "Education", "Vehicle", "Home", "Other"]
                    )
                    deadline = st.date_input("Target Date")

                submitted = st.form_submit_button("Create Goal", type="primary", use_container_width=True)

                if submitted and goal_name and target_amount > 0:
                    if add_goal(goal_name, target_amount, str(deadline), goal_category):
                        st.success(f"✅ Created goal: {goal_name}")
                        add_achievement("Goal Setter", "Created your first financial goal!")
                        st.rerun()

        with tab2:
            st.subheader("📈 Your Active Goals")

            goals_df = get_goals()

            if len(goals_df) > 0:
                for _, goal in goals_df.iterrows():
                    progress = (goal['current_amount'] / goal['target_amount']) * 100 if goal['target_amount'] > 0 else 0

                    with st.container():
                        st.write(f"### {goal['goal_name']}")

                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.progress(min(progress / 100, 1.0))
                            st.caption(f"₹{goal['current_amount']:,.2f} / ₹{goal['target_amount']:,.2f} ({progress:.1f}%)")

                        with col2:
                            st.metric("Remaining", f"₹{max(0, goal['target_amount'] - goal['current_amount']):,.2f}")

                        with col3:
                            st.write(f"**Deadline:** {goal['deadline']}")

                        col_a, col_b, col_c = st.columns([2, 1, 1])

                        with col_a:
                            add_amount = st.number_input(
                                "Add Amount",
                                min_value=0.0,
                                step=100.0,
                                key=f"add_{goal['id']}"
                            )

                        with col_b:
                            if st.button("💰 Add", key=f"btn_add_{goal['id']}"):
                                if add_amount > 0:
                                    if update_goal_progress(goal['id'], add_amount):
                                        st.success(f"Added ₹{add_amount:,.2f}!")
                                        st.rerun()

                        with col_c:
                            if st.button("🗑️ Delete", key=f"btn_del_{goal['id']}"):
                                if delete_goal(goal['id']):
                                    st.success("Goal deleted!")
                                    st.rerun()

                        st.divider()
            else:
                st.info("🎯 No active goals. Create one to start tracking your progress!")

        with tab3:
            st.subheader("🏆 Your Achievements")

            achievements_df = get_achievements()

            if len(achievements_df) > 0:
                cols = st.columns(3)
                for idx, achievement in achievements_df.iterrows():
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 20px; border-radius: 15px; text-align: center;
                                    margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>🏅</h2>
                            <h4 style='margin: 10px 0;'>{achievement['badge_name']}</h4>
                            <p style='margin: 5px 0; font-size: 0.9em;'>{achievement['description']}</p>
                            <small>{achievement['earned_date'][:10]}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🏆 No achievements yet. Keep using the app to earn badges!")

            st.divider()
            st.subheader("🔥 Your Saving Streak")
            streak = get_current_streak()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Streak", f"{streak} days")
            with col2:
                next_milestone = 7 if streak < 7 else 30 if streak < 30 else 100
                st.metric("Next Milestone", f"{next_milestone} days")
            with col3:
                st.metric("Days to Go", f"{max(0, next_milestone - streak)} days")

    # ===== ALERTS & NOTIFICATIONS PAGE =====
    elif page == "🔔 Alerts & Notifications":
        st.header("🔔 Budget Alerts & Notifications")

        tab1, tab2 = st.tabs(["➕ Set Alerts", "📊 Active Alerts"])

        with tab1:
            st.subheader("Create Budget Alert")
            st.write("Get notified when you're about to exceed your budget for a category.")

            with st.form("alert_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    alert_category = st.selectbox(
                        "Category",
                        ["Food", "Transport", "Entertainment", "Rent", "Utilities", "Shopping", "Healthcare", "Other"]
                    )

                with col2:
                    budget_limit = st.number_input("Monthly Budget (₹)", min_value=100.0, step=500.0, value=5000.0)

                with col3:
                    threshold = st.slider("Alert at % spent", 50, 100, 80)

                submitted = st.form_submit_button("Set Alert", type="primary", use_container_width=True)

                if submitted and budget_limit > 0:
                    if add_budget_alert(alert_category, budget_limit, threshold):
                        st.success(f"✅ Alert set for {alert_category} at {threshold}% of ₹{budget_limit:,.2f}")
                        st.rerun()

        with tab2:
            st.subheader("📋 Your Active Alerts")

            alerts_df = get_budget_alerts()

            if len(alerts_df) > 0:
                with sqlite3.connect('expenses.db') as conn:
                    spending_df = pd.read_sql("""
                        SELECT category, SUM(amount) as spent
                        FROM expenses
                        WHERE is_income = 0
                        AND date >= date('now', 'start of month')
                        GROUP BY category
                    """, conn)

                    for _, alert in alerts_df.iterrows():
                        category = alert['category']
                        limit_amt = float(alert['budget_limit'])
                        threshold = int(alert['alert_threshold'])

                        row = spending_df[spending_df['category'] == category]
                        spent = float(row['spent'].iloc[0]) if len(row) else 0.0
                        pct = (spent / limit_amt * 100) if limit_amt > 0 else 0.0
                        pct_clamped = max(0.0, min(100.0, pct))

                    with st.container():
                        st.write(f"### {category}")
                        st.progress(min(1.0, pct_clamped / 100.0))
                        st.caption(f"₹{spent:,.2f} / ₹{limit_amt:,.2f} ({pct:.1f}%)")

                        if pct >= 100:
                            st.error(f"🚨 Budget exceeded by ₹{spent - limit_amt:,.2f}")
                        elif pct >= threshold:
                            st.warning(f"⚠️ {threshold}% threshold reached")

                        cols = st.columns(2)
                        with cols[0]:
                            new_limit = st.number_input(
                                "Update Limit",
                                min_value=100.0,
                                step=500.0,
                                value=float(limit_amt),
                                key=f"upd_limit_{alert['id']}"
                            )
                        with cols[1]:
                            new_thresh = st.slider(
                                "Update Threshold",
                                50, 100, threshold, key=f"upd_thr_{alert['id']}"
                            )

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("💾 Save", key=f"save_alert_{alert['id']}"):
                                if add_budget_alert(category, new_limit, new_thresh):
                                    st.success("Alert updated")
                                    st.rerun()
                        with col_b:
                            if st.button("🗑️ Remove", key=f"del_alert_{alert['id']}"):
                                try:
                                    with sqlite3.connect('expenses.db') as conn:
                                        cur = conn.cursor()
                                        cur.execute("DELETE FROM budget_alerts WHERE id = ?", (int(alert['id']),))
                                        conn.commit()
                                    st.success("Alert removed")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error removing alert: {e}")
            else:
                st.info("No active alerts. Create one in Set Alerts tab.")

    # ===== AI ADVISOR PAGE =====
    elif page == "🤖 AI Advisor":
        st.header("🤖 AI Financial Advisor")
        st.caption("This guidance is informational and not professional financial advice.")

        question = st.text_input("Ask a financial question", placeholder="e.g., How can I reduce my monthly food spend?")
        if st.button("Get Advice", use_container_width=True) and question:
            with st.spinner("🧠 Thinking..."):
                advice = get_financial_advice(question)
                st.info(advice)

if __name__ == "__main__":
    main()

