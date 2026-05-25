import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# BƯỚC 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
df = pd.read_csv('cars_prediction/data/car_purchasing.csv')

print("First 5 rows of initial data:")
print(df.head(), "\n")


# ==========================================
# BƯỚC 2: TÁCH FEATURES (X) VÀ TARGET (y)
# ==========================================
X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']


# ==========================================
# BƯỚC 3: CHIA TẬP TRAIN / VALIDATION / TEST (80 - 10 - 10)
# ==========================================
# Lần cắt 1: Lấy 80% cho Train, 20% còn lại gộp chung vào tập Temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Lần cắt 2: Cắt đôi tập Temp (20%) thành 10% Validation và 10% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# ==========================================
# BƯỚC 4: CHUẨN HÓA DỮ LIỆU (NORMALIZATION)
# ==========================================
scaler_X = MinMaxScaler()

# 1. Nhớ kỹ: Chỉ Dùng fit_transform cho tập Train
X_train_scaled = scaler_X.fit_transform(X_train)

# 2. Dùng transform cho cả tập Validation và tập Test
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Chuẩn hóa cột đáp án (y) tương tự
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


# ==========================================
# IN KẾT QUẢ KIỂM TRA
# ==========================================
print("--- PREPROCESSING COMPLETED ---")
print(f"Number of customers for Train (80%): {X_train_scaled.shape[0]}")
print(f"Number of customers for Validation (10%): {X_val_scaled.shape[0]}")
print(f"Number of customers for Test (10%): {X_test_scaled.shape[0]}")
print("\nData of the first customer in the Train set after normalization:")
print(X_train_scaled[0])