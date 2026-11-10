import joblib
import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="RTO Risk Classifier API",
    description="Real-time COD RTO risk scoring engine preserving Colab pipeline integrity",
    version="1.0",
)

# Enable CORS so your frontend can communicate from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            return {
                "order_id": order.order_id,
                "risk_probability": 0.0,
                "risk_band": "LOW",
                "action": "Auto-Approve (Prepaid Order)"
            }
        X_input = extract_features(order)
        prob_rto = float(model.predict_proba(X_input)[0][1])
        
        if prob_rto < 0.3:
            risk_band, action = "LOW", "Auto-Approve"
        elif prob_rto <= 0.7:
            risk_band, action = "MEDIUM", "Trigger WhatsApp Confirmation Nudge"
        else:
            risk_band, action = "HIGH", "Trigger Prepaid Discount Nudge or Block"
            
        return {
            "order_id": order.order_id,
            "risk_probability": round(prob_rto, 4),
            "risk_band": risk_band,
            "action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "RTO Classifier API is online and synchronized!"}
