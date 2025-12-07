import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'

WEIGHTS_PATH = MODELS_DIR / 'logistic_weights.pkl'
SCALER_PATH = MODELS_DIR / 'minmax_scaler.pkl'

# --- 2. HÀM HỖ TRỢ ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@st.cache_resource
def load_resources():
    try:
        w = joblib.load(WEIGHTS_PATH)
        scaler = joblib.load(SCALER_PATH)
        return w, scaler
    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy file model ({e}). Hãy chạy train trước.")
        return None, None

# --- 3. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Airline Satisfaction Prediction", layout="wide")
st.title("✈️ Dự đoán mức độ hài lòng hành khách")
st.markdown("---")

w, scaler = load_resources()

if w is not None and scaler is not None:
    
    with st.form("input_form"):
        st.header("1. Nhập thông tin hành khách")
        col1, col2, col3 = st.columns(3)
        
        # --- Cột 1: Thông tin cá nhân & Chuyến bay ---
        with col1:
            st.subheader("Thông tin chung")
            # LabelEncoder: Female=0, Male=1
            gender = st.selectbox("Giới tính", ["Female", "Male"])
            
            # LabelEncoder: Loyal Customer=0, disloyal Customer=1 (L < d)
            cust_type = st.selectbox("Loại khách hàng", ["Loyal Customer", "disloyal Customer"])
            
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=30)
            
            # LabelEncoder: Business travel=0, Personal Travel=1
            travel_type = st.selectbox("Mục đích chuyến đi", ["Business travel", "Personal Travel"])
            
            # LabelEncoder: Business=0, Eco=1, Eco Plus=2
            travel_class = st.selectbox("Hạng vé", ["Business", "Eco", "Eco Plus"])
            
            distance = st.number_input("Khoảng cách bay (km)", min_value=0, value=500)
            
            dep_delay = st.number_input("Trễ khởi hành (phút)", min_value=0, value=0)
            arr_delay = st.number_input("Trễ đến nơi (phút)", min_value=0, value=0)

        # --- Cột 2: Dịch vụ trực tuyến & Checkin ---
        with col2:
            st.subheader("Đánh giá dịch vụ (0-5)")
            wifi = st.slider("Inflight wifi service", 0, 5, 3)
            time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
            booking = st.slider("Ease of Online booking", 0, 5, 3)
            gate = st.slider("Gate location", 0, 5, 3)
            food = st.slider("Food and drink", 0, 5, 3)
            boarding = st.slider("Online boarding", 0, 5, 3)
            seat = st.slider("Seat comfort", 0, 5, 3)
            entertainment = st.slider("Inflight entertainment", 0, 5, 3)

        # --- Cột 3: Dịch vụ trên máy bay ---
        with col3:
            st.subheader("Dịch vụ trên máy bay")
            onboard_svc = st.slider("On-board service", 0, 5, 3)
            leg_room = st.slider("Leg room service", 0, 5, 3)
            baggage = st.slider("Baggage handling", 0, 5, 3)
            checkin = st.slider("Checkin service", 0, 5, 3)
            inflight_svc = st.slider("Inflight service", 0, 5, 3)
            cleanliness = st.slider("Cleanliness", 0, 5, 3)

        submit_btn = st.form_submit_button("🔍 Dự đoán ngay")

    # --- Xử lý dự đoán ---
    if submit_btn:
        st.markdown("---")
        st.header("2. Kết quả")
        
        # Mapping dữ liệu thủ công để khớp với LabelEncoder lúc train
        # Cần chú ý thứ tự sort ABC của LabelEncoder
        
        val_gender = 0 if gender == "Female" else 1
        val_cust = 0 if cust_type == "Loyal Customer" else 1
        val_travel = 0 if travel_type == "Business travel" else 1
        
        if travel_class == "Business": val_class = 0
        elif travel_class == "Eco": val_class = 1
        else: val_class = 2 # Eco Plus
        
        # Tạo vector input theo đúng thứ tự cột trong file CSV (trừ id, Unnamed:0, satisfaction)
        # Thứ tự chuẩn:
        # [Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, 
        # Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, 
        # Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, 
        # On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, 
        # Cleanliness, Departure Delay in Minutes, Arrival Delay in Minutes]
        
        input_data = [
            val_gender, val_cust, age, val_travel, val_class, distance,
            wifi, time_conv, booking, gate, food, boarding, seat, entertainment,
            onboard_svc, leg_room, baggage, checkin, inflight_svc, cleanliness,
            dep_delay, arr_delay
        ]
        
        X_input = np.array(input_data).reshape(1, -1)
        
        # Scale dữ liệu
        try:
            X_scaled = scaler.transform(X_input)
            
            # Tính toán sigmoid(X * w)
            z = np.dot(X_scaled, w)
            prob = sigmoid(z)[0][0]
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prob >= 0.5:
                    st.success("## 😊 HÀI LÒNG")
                    st.write("Khách hàng có khả năng cao sẽ hài lòng với dịch vụ.")
                else:
                    st.error("## 😞 KHÔNG HÀI LÒNG")
                    st.write("Khách hàng có nguy cơ không hài lòng.")
                st.metric("Xác suất hài lòng", f"{prob*100:.1f}%")

            with col_res2:
                # Hiển thị biểu đồ đóng góp (Feature Contribution) cho từng dự đoán cụ thể
                # Contribution = Feature_Value_Scaled * Weight
                contribution = X_scaled[0] * w.flatten()
                
                # Top 5 yếu tố ảnh hưởng tích cực nhất và tiêu cực nhất
                features_list = [
                    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
                    'Wifi', 'Time Convenient', 'Online Booking', 'Gate Location', 'Food/Drink', 
                    'Online Boarding', 'Seat Comfort', 'Entertainment', 'On-board Svc', 
                    'Leg Room', 'Baggage', 'Checkin', 'Inflight Svc', 'Cleanliness', 
                    'Dep Delay', 'Arr Delay'
                ]
                
                df_contrib = pd.DataFrame({
                    'Feature': features_list,
                    'Contribution': contribution
                }).sort_values(by='Contribution', ascending=False)
                
                st.write("### Yếu tố ảnh hưởng chính đến kết quả này")
                st.bar_chart(df_contrib.set_index('Feature').head(7)) # Top 7 yếu tố tích cực
                
        except Exception as e:
            st.error(f"Lỗi tính toán: {e}")
            st.write("Kiểm tra lại số lượng features đầu vào.")

    # --- Phần hiển thị thông tin Weights (Global) ---
    st.markdown("---")
    with st.expander("Xem chi tiết trọng số mô hình (Model Weights)"):
        st.write("Biểu đồ này thể hiện mức độ quan trọng tổng quát của từng đặc trưng mà mô hình đã học được.")
        
        features_list = [
            'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
            'Wifi', 'Time Convenient', 'Online Booking', 'Gate Location', 'Food/Drink', 
            'Online Boarding', 'Seat Comfort', 'Entertainment', 'On-board Svc', 
            'Leg Room', 'Baggage', 'Checkin', 'Inflight Svc', 'Cleanliness', 
            'Dep Delay', 'Arr Delay'
        ]
        
        if len(w.flatten()) == len(features_list):
            df_weights = pd.DataFrame({
                'Feature': features_list,
                'Weight': w.flatten()
            }).sort_values(by='Weight')
            
            st.bar_chart(df_weights, x='Feature', y='Weight')
        else:
            st.warning("Số lượng weights không khớp với danh sách feature hiển thị.")