import pandas as pd

# 1. تحميل البيانات
# تأكدي أن ملف CSV موجود في نفس المجلد مع ملف الكود
file_path = 'MASTER_DATABASE_FINAL.csv'
df = pd.read_csv(file_path)

# 2. عرض أول 5 أسطر للتأكد من سلامة البيانات
print("--- الخمسة أسطر الأولى من البيانات ---")
print(df.head())

# 3. عرض أسماء الأعمدة وأنواع البيانات
print("\n--- أسماء الأعمدة ونوع البيانات في كل عمود ---")
print(df.info())

# 4. التحقق من القيم المفقودة (Null values)
print("\n--- عدد القيم المفقودة في كل عمود ---")
print(df.isnull().sum())