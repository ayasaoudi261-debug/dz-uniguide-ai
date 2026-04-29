import pandas as pd

# 1. قاموس الكلمات المفتاحية (يمكننا توسيعه بناءً على الاستبيان)
keywords_map = {
    'computer': ['INFORMATIQUE', 'INTELLIGENCE ARTIFICIELLE', 'SYSTEMES INDOSTRY'],
    'برمجة': ['INFORMATIQUE', 'INTELLIGENCE ARTIFICIELLE'],
    'طب': ['MEDECINE', 'PHARMACIE', 'CHIRURGIE DENTAIRE', 'SANTÉ'],
    'health': ['MEDECINE', 'PHARMACIE', 'SANTÉ'],
    'لغات': ['ANGLAIS', 'FRANCAIS', 'TRADUCTION'],
    'english': ['ANGLAIS', 'TRADUCTION'],
    'اقتصاد': ['ECONOMIE', 'GESTION', 'COMMERCE'],
    'تكنولوجيا': ['TECHNOLOGIE', 'GENIE', 'AERONAUTIQUE']
}

def extract_interests(user_text):
    """دالة لاستخراج التخصصات المرتبطة بكلام الطالب"""
    user_text = user_text.lower()
    found_specialties = []
    
    for key, specialties in keywords_map.items():
        if key in user_text:
            found_specialties.extend(specialties)
    
    return list(set(found_specialties)) # حذف التكرار

# 2. تجربة المحرك
test_sentence = "أريد دراسة تخصص له علاقة بالبرمجة و computer"
interests = extract_interests(test_sentence)

print(f"النص المدخل: {test_sentence}")
print(f"التخصصات المقترحة بناءً على الاهتمامات: {interests}")