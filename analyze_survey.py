import pandas as pd
import os

# اسم الملف كما ذكرتِ تماماً
file_name = 'data_set_PFE_m2 (réponses).xlsx'

if not os.path.exists(file_name):
    print(f"خطأ: لم يتم العثور على الملف '{file_name}'. تأكدي من وجوده في: {os.getcwd()}")
else:
    try:
        # قراءة ملف الإكسل
        survey_df = pd.read_excel(file_name)
        print(f"✅ تم تحميل الملف بنجاح!")
        
        # عرض الأعمدة
        print("\n--- أعمدة الاستبيان ---")
        print(survey_df.columns.tolist())

        # عرض أهم الاهتمامات الأكاديمية
        # ملاحظة: تأكدي أن اسم العمود 'academic_interests' مطابق لما في ملفك
        if 'academic_interests' in survey_df.columns:
            print("\n--- أعلى 10 اهتمامات أكاديمية ---")
            print(survey_df['academic_interests'].value_counts().head(10))
        
        # عرض عينة من الأسئلة
        if 'user_question' in survey_df.columns:
            print("\n--- عينة من أسئلة الطلاب ---")
            print(survey_df['user_question'].dropna().head(5))

    except Exception as e:
        print(f"حدث خطأ أثناء قراءة الملف: {e}")