import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib

# ======================================
# 1️⃣ تحميل البيانات
# ======================================
df = pd.read_excel('data_set_PFE_m2 (réponses).xlsx')

# ======================================
# 2️⃣ تنظيف وتحويل البيانات
# ======================================

# Encoding للشعبة
le_stream = LabelEncoder()
df['bac_stream_encoded'] = le_stream.fit_transform(df['bac_stream'].astype(str))

# Encoding للعوامل
le_factor = LabelEncoder()
df['factor1_encoded'] = le_factor.fit_transform(df['factor_1_importance'].astype(str))

# Encoding للميول (التخصص)
le_major = LabelEncoder()
df['academic_interests'] = df['academic_interests'].astype(str)
le_major.fit(df['academic_interests'])

# استخراج المعدل (يدعم 15.5)
df['avg_numeric'] = df['bac_average'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# حذف القيم الناقصة
df = df.dropna(subset=['avg_numeric'])

# ======================================
# 🎯 3️⃣ إنشاء Dataset ذكية (Student vs Major)
# ======================================

all_majors = df['academic_interests'].unique()
new_data = []

for _, row in df.iterrows():
    for major in all_majors:
        new_row = {}

        # بيانات الطالب
        new_row['bac_stream_encoded'] = row['bac_stream_encoded']
        new_row['avg_numeric'] = row['avg_numeric']
        new_row['factor1_encoded'] = row['factor1_encoded']

        # التخصص
        new_row['major_encoded'] = le_major.transform([major])[0]

        # 🎯 target (هل هذا تخصص الطالب الحقيقي؟)
        if major == row['academic_interests']:
            new_row['target'] = 1
        else:
            new_row['target'] = 0

        new_data.append(new_row)

df_model = pd.DataFrame(new_data)

# ======================================
# 4️⃣ اختيار Features
# ======================================
X = df_model[['bac_stream_encoded', 'avg_numeric', 'factor1_encoded', 'major_encoded']]
y = df_model['target']

# ======================================
# 5️⃣ موازنة البيانات
# ======================================
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# ======================================
# 6️⃣ تقسيم البيانات
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ======================================
# 7️⃣ تدريب النماذج
# ======================================

# 🌳 Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# 🚀 Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# ======================================
# 8️⃣ التقييم
# ======================================

# Random Forest
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n🌳 Random Forest Results:")
print("Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

# Gradient Boosting
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

print("\n🚀 Gradient Boosting Results:")
print("Accuracy:", gb_acc)
print(classification_report(y_test, gb_pred))

# ======================================
# 9️⃣ اختيار الأفضل
# ======================================
if rf_acc > gb_acc:
    best_model = rf_model
    model_name = "Random Forest"
else:
    best_model = gb_model
    model_name = "Gradient Boosting"

print(f"\n🏆 أفضل موديل: {model_name}")

# ======================================
# 🔟 حفظ كل شيء
# ======================================
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(le_stream, 'stream_encoder.pkl')
joblib.dump(le_factor, 'factor_encoder.pkl')
joblib.dump(le_major, 'major_encoder.pkl')

print("✅ تم حفظ الموديل و جميع الـ encoders بنجاح!")