import joblib
import pandas as pd
import re
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="RTO Risk Classifier API",
    description="Real-time COD RTO risk scoring engine with analytics logging",
    version="1.1",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite Database for Analytics Logging
def init_db():
    conn = sqlite3.connect("rto_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            order_id TEXT,
            order_amount REAL,
            payment_method TEXT,
            shipping_pincode INTEGER,
            shipping_address TEXT,
            risk_probability REAL,
            risk_band TEXT,
            action TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load model artifacts on startup
try:
    model = joblib.load("rto_logreg.joblib")
    meta = joblib.load("rto_meta.joblib")
    pin_table = meta["pin_table"]
    global_rate = meta["global_rate"]
    feature_list = meta["features"]
    print("Model and metadata artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

class OrderPayload(BaseModel):
    order_id: str
    order_amount: float
    payment_method: str
    shipping_pincode: int
    shipping_address: str
    order_hour: int

def extract_features(data: OrderPayload):
    address = str(data.shipping_address)
    addr_len = len(address)
    addr_words = len(address.split())
    has_missing_number = 1 if not re.search(r'\b\d+([/-]\d+)?\b', address) else 0
    has_filler = 1 if any(word in address.lower() for word in ["test", "near", "xyz", "abc", "address"]) else 0
    pin = int(data.shipping_pincode)
    pin_rto_rate = pin_table.get(pin, global_rate)
    
    features_dict = {
        "order_amount": data.order_amount,
        "order_hour": data.order_hour,
        "addr_len": addr_len,
        "addr_words": addr_words,
        "has_missing_number": has_missing_number,
        "has_filler": has_filler,
        "pin_rto_rate": pin_rto_rate
    }
    return pd.DataFrame([features_dict])[feature_list]

@app.post("/predict")
def predict_rto(order: OrderPayload):
    try:
        if order.payment_method.lower() == "prepaid":
            risk_probability = 0.0
            risk_band = "LOW"
            action = "Auto-Approve (Prepaid Order)"
        else:
            X_input = extract_features(order)
            prob_rto = float(model.predict_proba(X_input)[0][1])
            risk_probability = round(prob_rto, 4)
            
            if prob_rto < 0.3:
                risk_band, action = "LOW", "Auto-Approve"
            elif prob_rto <= 0.7:
                risk_band, action = "MEDIUM", "Trigger WhatsApp Confirmation Nudge"
            else:
                risk_band, action = "HIGH", "Trigger Prepaid Discount Nudge or Block"

        # Log prediction to SQLite database
        conn = sqlite3.connect("rto_logs.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (timestamp, order_id, order_amount, payment_method, shipping_pincode, shipping_address, risk_probability, risk_band, action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            order.order_id,
            order.order_amount,
            order.payment_method,
            order.shipping_pincode,
            order.shipping_address,
            risk_probability,
            risk_band,
            action
        ))
        conn.commit()
        conn.close()

        return {
            "order_id": order.order_id,
            "risk_probability": risk_probability,
            "risk_band": risk_band,
            "action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    """Endpoint to view prediction logs and summary stats"""
    conn = sqlite3.connect("rto_logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(risk_probability) FROM predictions")
    total_orders, avg_risk = cursor.fetchone()
    
    cursor.execute("SELECT risk_band, COUNT(*) FROM predictions GROUP BY risk_band")
    bands = dict(cursor.fetchall())
    
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    recent_logs = cursor.fetchall()
    conn.close()
    
    return {
        "total_scored_orders": total_orders or 0,
        "average_risk_probability": round(avg_risk or 0.0, 4),
        "risk_breakdown": bands,
        "recent_predictions": recent_logs
    }

@app.get("/")
def home():
    return {"status": "RTO Classifier API with Analytics is online and synchronized!"}

