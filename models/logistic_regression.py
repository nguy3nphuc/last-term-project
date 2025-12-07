import numpy as np
from pathlib import Path
from data.data import load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
import traceback
import pandas as pd

# 1. Xác định thư mục gốc của dự án
# Lùi lại 2 cấp từ models/logistic_regression.py -> do_an_cuoi_ky/
ROOT_DIR = Path(__file__).resolve().parents[1]

# 2. Định nghĩa đường dẫn tuyệt đối đến file
MODELS_DIR = ROOT_DIR / 'models'
WEIGHTS_PATH = MODELS_DIR / 'logistic_weights.pkl'
SCALER_PATH = MODELS_DIR / 'minmax_scaler.pkl'

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, w, eta):
    for _ in range(1000):
        for i in range(X.shape[0]):
            xi = X[i].reshape(-1, 1)
            yi = y[i]
            pi = sigmoid(w.T @ xi)
            w -= eta * (pi - yi) * xi
    return w

def train_logistic_regression():
    # 1. Tải và xử lý dữ liệu bằng hàm đã import
    try:
        df = load_and_preprocess_data()
    except Exception:
        print("Lỗi khi load dữ liệu:")
        traceback.print_exc()
        return
    
    # 2. Chuẩn bị dữ liệu và huấn luyện model
    # ... code chia tập X, y, và fit model
    X = df.drop(columns="satisfaction", axis=1)
    y = df['satisfaction']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=42)
    # Đảm bảo y là mảng số (0/1). Nếu chưa phải numeric, cố gắng ép kiểu.
    # if not np.issubdtype(y.dtype, np.number):
    #     try:
    #         y = y.astype(int)
    #     except Exception:
    #         # As a fallback, use LabelEncoder-like mapping
    #         uniques = list(pd.Series(y).unique())
    #         mapping = {val: idx for idx, val in enumerate(uniques)}
    #         y = pd.Series(y).map(mapping).astype(int)

    num_features = X_train.shape[1] 
    w = np.random.randn(num_features, 1)
    eta = 0.01

    try:
        w = gradient_descent(X_train, y_train.values, w, eta)

        y_pred_gd = sigmoid(X_test @ w).flatten()
        y_pred_gd = (y_pred_gd >= 0.5).astype(int)

        # In ra độ chính xác của mô hình (y_true, y_pred)
        print('Accuracy (Gradient Descent): ', accuracy_score(y_test, y_pred_gd))
    except Exception:
        print("Lỗi trong quá trình huấn luyện/dự đoán:")
        traceback.print_exc()
        return

    # ... logic huấn luyện model ...
    print("Dữ liệu đã được tải và xử lý thành công!")

    # Lưu trọng số w vào file
    try:
        saved = joblib.dump(w, str(WEIGHTS_PATH))
        print(f"Lưu thành công w tại: {saved}")
    except Exception:
        print("Không thể lưu weights:")
        traceback.print_exc()

    # Lưu đối tượng Scaler
    try:
        saved_scaler = joblib.dump(scaler, str(SCALER_PATH))
        print(f"Lưu thành công scaler tại: {saved_scaler}")
    except Exception:
        print("Không thể lưu scaler:")
        traceback.print_exc()

if __name__ == '__main__':
    train_logistic_regression()