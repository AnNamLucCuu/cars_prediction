import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



# BƯỚC 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# Đọc file dữ liệu 
df = pd.read_csv('cars_prediction/data/car_purchasing.csv')

# Xem trước 5 dòng đầu tiên
print("First 5 rows of initial data :")
print(df.head(), "\n")



# BƯỚC 2: TÁCH FEATURES (X) VÀ TARGET (y)
# X: Lấy tất cả các cột trừ cột đáp án 'car purchase amount'
X = df.drop('car purchase amount', axis=1)

# y: Chỉ lấy cột đáp án
y = df['car purchase amount']




# BƯỚC 3: CHIA TẬP TRAIN / TEST
# Luôn chia Train/Test TRƯỚC KHI chuẩn hóa để tránh rò rỉ dữ liệu (Data Leakage)
# test_size=0.2: Trích ra 20% dữ liệu để thi (Test), 80% để học (Train)
# random_state=42: Đảm bảo kết quả chia ngẫu nhiên giống hệt nhau ở mọi lần chạy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# BƯỚC 4: CHUẨN HÓA DỮ LIỆU (NORMALIZATION)
# Khởi tạo công cụ chuẩn hóa (đưa số liệu về khoảng 0 - 1)
scaler_X = MinMaxScaler()

# 1. Dùng fit_transform cho tập Train: 
# Máy sẽ tìm Min, Max của tập Train và tự động thu nhỏ các số liệu lại
X_train_scaled = scaler_X.fit_transform(X_train)

# 2. Chỉ dùng transform cho tập Test: 
# Bắt buộc phải dùng lại cái "thước đo" (Min, Max) đã học ở tập Train để áp dụng cho tập Test. 
X_test_scaled = scaler_X.transform(X_test)

# (Tùy chọn) Đối với bài toán Hồi quy dùng Mạng Nơ-ron, ta thường chuẩn hóa cả cột đáp án (y)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


# IN KẾT QUẢ KIỂM TRA
print("--- PREPROCESSING COMPLETED ---")
print(f"Number of customers for Train (80%): {X_train_scaled.shape[0]} ")
print(f"Number of customers for Test (20%): {X_test_scaled.shape[0]} ")
print("\nData of the first customer in the Train set after normalization:")
print(X_train_scaled[0]) # Các con số khổng lồ (60,000) giờ đã nằm trong khoảng 0 -> 1



