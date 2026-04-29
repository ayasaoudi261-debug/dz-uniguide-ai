import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# تحميل البيانات
df = pd.read_csv('MASTER_DATABASE_FINAL.csv')

# ======================================
# 🎯 Stream Encoder
# ======================================
stream_encoder = LabelEncoder()
df['stream_encoded'] = stream_encoder.fit_transform(df['Stream'].astype(str))

with open('stream_encoder.pkl', 'wb') as f:
    pickle.dump(stream_encoder, f)

# ======================================
# 🎯 Major Encoder
# ======================================
major_encoder = LabelEncoder()
df['major_encoded'] = major_encoder.fit_transform(df['University_Major'].astype(str))

with open('major_encoder.pkl', 'wb') as f:
    pickle.dump(major_encoder, f)

print("✅ تم إنشاء جميع الـ encoders بنجاح")