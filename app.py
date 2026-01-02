from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
from model import TempLSTM

app = Flask(__name__)

# ---------------- Load model and stats ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TempLSTM()
model.load_state_dict(torch.load("temp_lstm_model.pth", map_location=device))
model.to(device)
model.eval()

train_mean = np.load("train_mean.npy")
train_std = np.load("train_std.npy")

# Load dataset for last 7 days
df = pd.read_csv("constantine_temp_2010_2025.csv", parse_dates=['date'])

SEQ_LENGTH = 7

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        date_str = request.form.get("date")
        try:
            target_date = pd.to_datetime(date_str)
            mask = (df['date'] < target_date) & (df['date'] >= target_date - pd.Timedelta(days=SEQ_LENGTH))
            last_7_days = df.loc[mask, 'temp'].values

            if len(last_7_days) < SEQ_LENGTH:
                prediction = "Not enough data to predict."
            else:
                X_input = (last_7_days - train_mean) / train_std
                X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                with torch.no_grad():
                    pred = model(X_input).cpu().numpy()[0]
                prediction = f"{pred:.2f} °C"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
