import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# تحميل البيانات
df = pd.read_csv('MASTER_DATABASE_FINAL.csv')

# إنشاء encoder للتخصصات
major_encoder = LabelEncoder()
df['major_encoded'] = major_encoder.fit_transform(df['University_Major'].astype(str))

# حفظه
with open('major_encoder.pkl', 'wb') as f:
    pickle.dump(major_encoder, f)

print("✅ تم إنشاء major_encoder.pkl بنجاح")