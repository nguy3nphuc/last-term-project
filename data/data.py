import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Định nghĩa đường dẫn gốc (root directory) của dự án
# Dùng Path(__file__).resolve().parents[1] để nhảy từ data/data.py (parents[0])
# lên thư mục gốc do_an_cuoi_ky (parents[1])
ROOT_DIR = Path(__file__).resolve().parents[1]

def load_and_preprocess_data():
    # Xây dựng đường dẫn tuyệt đối đến file test.csv
    data_file_path = ROOT_DIR / 'test.csv'
    
    # Kiểm tra xem file có tồn tại không
    if not data_file_path.exists():
        raise FileNotFoundError(f"File data không tìm thấy tại: {data_file_path}")

    # Load dataset
    df = pd.read_csv(data_file_path) 
    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 0', 'id'], inplace=True, errors='ignore')
    # Drop rows with missing values
    df = df.dropna(axis=0)
    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype != np.number:
            df[column] = le.fit_transform(df[column])
            
    return df