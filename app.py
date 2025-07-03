from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading Trained and saved Randomforest, DNN and CNN models with formats .pkl and .keras
rf_binary = joblib.load("rf_binary.pkl")
rf_mc = joblib.load("rf_multi.pkl")
dnn_binary = tf.keras.models.load_model("dnn_binary.keras")
dnn_mc = tf.keras.models.load_model("dnn_multi.keras")
cnn_binary = tf.keras.models.load_model("cnn_binary.keras")
cnn_mc = tf.keras.models.load_model("cnn_multi.keras")

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df = df.head(100)  # Limit to 100 rows for performance
        df = df.drop(columns=["id", "attack_cat", "label", "rate"], errors="ignore")  # Drop unnecessary columns
        df = df.rename(
            columns={
                "spkts": "Spkts",
                "dpkts": "Dpkts",
                "sload": "Sload",
                "dload": "Dload",
                "djit": "Djit",
                "sjit": "Sjit",
                "smean": "smeansz",
                "dmean": "dmeansz",
                "ct_src_ltm": "ct_src_ ltm",
                "response_body_len": "res_bdy_len",
                "sinpkt": "Sintpkt",
                "dinpkt": "Dintpkt",
            }
        )
        # Strip whitespace and replace blank strings with NaN
        df["ct_ftp_cmd"] = df["ct_ftp_cmd"].astype(str).str.strip().replace("", None)
        
        # Convert to numeric, coerce errors to NaN, then fill with 0 and convert to int
        df["ct_ftp_cmd"] = (pd.to_numeric(df["ct_ftp_cmd"], errors="coerce").fillna(0).astype(int))

        cols_to_drop = ["sloss", "Dpkts", "dloss","synack","ackdat","dwin","Dintpkt",
                        "ct_dst_sport_ltm","ct_ftp_cmd" ]
        df.drop(labels=cols_to_drop, axis=1, inplace=True)

        # Label encoding for categorical features
        label_encoders = {}  # To store encoders for each column
        for col in df.columns:
            if df[col].dtypes == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        # PCA transformation
        pca = PCA(n_components=25) # Add svd_solver='full' if needed
        df_pca = pca.fit_transform(df)
        
        # Pad with one 0 to make it 38, which can be reshaped to 6x7
        def reshape_to_image(X, h=6, w=7):
            padded = np.pad(X.shape, ((0, 0), (0, h * w - X.shape[1])), 'constant')  # zero padding
            return padded.reshape(-1, h, w, 1)  # grayscale image shape
        df_reshaped = reshape_to_image(df, 6, 7)
        
        class_names = ['DoS', 'DoS/DDoS', 'Exploits', 'Fuzzing', 'Malware', 'Normal Traffic', 'Reconnaissance']
        class_name = ['Normal Traffic', 'Attack']
        
        # Predictions
        rf_binary_preds = [class_name[i] for i in rf_binary.predict(df_pca)]
        rf_mc_preds = [class_names[i] for i in rf_mc.predict(df_pca)]
        
        # rf_binary_preds = rf_binary.predict(df_pca)
        # rf_mc_preds = rf_mc.predict(df_pca)
        dnn_binary_preds = dnn_binary.predict(df_pca)
        dnn_mc_preds = dnn_mc.predict(df_pca)
        cnn_binary_preds = cnn_binary.predict(df_reshaped)
        cnn_mc_preds = cnn_mc.predict(df_reshaped)
        
        def reshape_predictions(preds, label):
            predicted_values = []
            for i in range(len(preds)):
                predicted_values.append(label[int(np.argmax(preds[i]))])
            return predicted_values
        
        dnn_b_preds = reshape_predictions(dnn_binary_preds, class_name)
        dnn_m_preds = reshape_predictions(dnn_mc_preds, class_names)
        cnn_b_preds = reshape_predictions(cnn_binary_preds, class_name)
        cnn_m_preds = reshape_predictions(cnn_mc_preds, class_names)
    
        return JSONResponse(
            content={
                "rf_binary_preds": rf_binary_preds,
                "rf_multi_preds": rf_mc_preds,
                "dnn_binary_preds": dnn_b_preds,
                "dnn_multi_preds": dnn_m_preds,
                "cnn_binary_preds": cnn_b_preds,
                "cnn_multi_preds": cnn_m_preds,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"An error occurred: {str(e)}"}
        )

