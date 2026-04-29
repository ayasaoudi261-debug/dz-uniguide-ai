import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib

# 1. تحميل وتجهيز البيانات
df = pd.read_excel('data_set_PFE_m2 (réponses).xlsx')

# تحويل البيانات النصية إلى أرقام بشكل احترافي
le_stream = LabelEncoder()
df['bac_stream_encoded'] = le_stream.fit_transform(df['bac_stream'].astype(str))

# استخراج المعدل بدقة
df['avg_numeric'] = df['bac_average'].str.extract('(\d+)').astype(float)

# الهدف: النجاح الأكاديمي
df['target'] = df['academic_status'].apply(lambda x: 1 if 'Master' in str(x) else 0)

# 2. مصفوفة الميزات X والهدف y
X = df[['bac_stream_encoded', 'avg_numeric']].fillna(0)
y = df['target']

# 3. موازنة البيانات (Oversampling) - هذا هو "سر" الذكاء هنا
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 4. تقسيم البيانات بعد الموازنة
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. تدريب الموديل (Random Forest) مع ضبط المعايير
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 6. التقييم
y_pred = model.predict(X_test)
print("--- تقرير أداء الموديل الذكي بعد الموازنة ---")
print(classification_report(y_test, y_pred))

# حفظ الموديل والمحول (Encoder) لاستخدامهما في Streamlit
joblib.dump(model, 'university_model_smart.pkl')
joblib.dump(le_stream, 'stream_encoder.pkl')
print("✅ تم حفظ الموديل 'العبقري' بنجاح!")