import pandas as pd

def calculate_weighted_average(bac_stream, m_bac, maths, physics, science, arabic, foreign_lang):
    """
    دالة لحساب المعدل الموزون بناءً على قوانين المنشور الوزاري الجزائري.
    هذه أمثلة وسنطورها بناءً على ملف circulaire-2025.
    """
    if bac_stream == "Science":
        # مثال: تخصصات العلوم الطبية (معدل البكالوريا * 2 + علوم) / 3
        return (m_bac * 2 + science) / 3
    elif bac_stream == "Mathematics":
        # مثال: تخصصات التكنولوجيا (معدل البكالوريا * 2 + رياضيات) / 3
        return (m_bac * 2 + maths) / 3
    else:
        return m_bac # للتبسيط حالياً

# تحميل قاعدة البيانات الأساسية
db_path = 'MASTER_DATABASE_FINAL.csv'
df_master = pd.read_csv(db_path)

def get_recommendations(student_stream, weighted_avg, wilaya):
    """
    تصفية التخصصات بناءً على:
    1. الشعبة (Stream)
    2. المعدل الموزون (أقل من أو يساوي معدل القبول)
    3. الولاية (Allowed_Wilayas)
    """
    
    # تحويل اسم العمود حسب الملف الخاص بك
    stream_col = f"Min_{student_stream}_x" # مثلا Min_Science_x
    
    if stream_col not in df_master.columns:
        return "هذه الشعبة غير موجودة في البيانات."

    # تصفية التخصصات التي تقبل هذا المعدل
    # نختار التخصصات التي معدل قبولها (Min) أصغر من أو يساوي معدل الطالب
    recommendations = df_master[
        (df_master[stream_col] > 0) & 
        (df_master[stream_col] <= weighted_avg)
    ]
    
    # تصفية إضافية حسب الولاية (National أو رقم الولاية)
    # ملاحظة: سنقوم بتحسين هذا الجزء لاحقاً ليكون أدق
    recommendations = recommendations[
        (recommendations['Allowed_Wilayas'].str.contains(str(wilaya))) | 
        (recommendations['Allowed_Wilayas'].str.contains('National'))
    ]
    
    return recommendations[['University_Major', 'University_Name', stream_col]].head(10)

# تجربة سريعة (Testing)
print("--- تجربة نظام التوصية الأولي ---")
# لنفترض طالب علوم، معدله 15، ونقطة العلوم 16، من ولاية 13
avg = calculate_weighted_average("Science", 15, 14, 14, 16, 12, 12)
results = get_recommendations("Science", avg, "13")
print(results)