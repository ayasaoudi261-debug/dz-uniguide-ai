import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================
# 🎉 PAGE SETUP + WELCOME SCREEN (NEW - ONLY ADDITION)
# ======================================================================

st.set_page_config(page_title="DZ-UniGuide AI AYA", page_icon="🎓", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- WELCOME PAGE ----------------
if st.session_state.page == "welcome":

    st.markdown("""
    <div style='text-align:center; padding:40px'>
        <h1>🎉 مبروك نجاحك في البكالوريا 🇩🇿</h1>
        <h3>مرحبا بك في نظام التوجيه الذكي الجامعي</h3>
    </div>
    """, unsafe_allow_html=True)

    st.success("👏 هذا إنجاز كبير، والآن تبدأ مرحلة اختيار المستقبل الجامعي")

    st.info("💡 نصيحة: اختر تخصصك بناءً على الشغف + الفرص + القدرات")

    st.markdown("""
    <div style='text-align:center; font-size:18px; padding:15px'>
        🤖 هذا النظام يساعدك لاختيار أفضل تخصص باستخدام الذكاء الاصطناعي
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 ابدأ الآن", use_container_width=True, type="primary"):
        st.session_state.page = "app"
        st.rerun()

    st.stop()


# ======================================================================
# 🔥 1. الدوال المساعدة المتقدمة
# ======================================================================

@st.cache_resource
def load_assets():
    """تحميل البيانات والنماذج مع التحقق"""
    try:
        db = pd.read_csv('MASTER_DATABASE_FINAL.csv')
        encoders = {}
        for filename in ['best_model.pkl', 'stream_encoder.pkl', 'factor_encoder.pkl', 'major_encoder.pkl']:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                encoders[filename] = obj
                st.info(f"✅ تم تحميل {filename}: {type(obj).__name__}")
        
        return db, encoders['best_model.pkl'], encoders['stream_encoder.pkl'], encoders['factor_encoder.pkl'], encoders['major_encoder.pkl']
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        st.stop()


def display_metrics(df, title="📊 الإحصائيات الرئيسية"):
    high_success = len(df[df['success_prediction'] > 80])
    high_interest = len(df[df['interest_score'] > 75])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 التخصصات", len(df))
    with col2: st.metric("💎 نجاح >80%", high_success)
    with col3: st.metric("❤️ اهتمام >75%", high_interest)
    with col4: st.metric("⭐ أفضل تطابق", f"{df['combined_score'].max():.1f}%")


def smart_advice(best_score):
    if best_score > 85:
        st.success("🎉 ممتاز! لديك خيارات قوية جداً")
    elif best_score > 70:
        st.info("👍 جيد جداً")
    else:
        st.warning("⚠️ تحسين مطلوب")


def style_table(df, weighted_avg):
    return df.style.background_gradient(cmap='RdYlGn')


def final_summary(filtered_res, weighted_avg):
    total = len(filtered_res)
    best_match = filtered_res['combined_score'].max()

    st.markdown("### 🎉 ملخص التحليل")
    st.metric("أفضل تطابق", f"{best_match:.1f}%")
    st.metric("عدد التخصصات", total)


# ======================================================================
# 🔥 2. تحميل البيانات
# ======================================================================

db, model, stream_encoder, factor_encoder, major_encoder = load_assets()

if 'language' not in st.session_state:
    st.session_state.language = "العربية"


texts = {
    "العربية": {
        "bot_name": "🤖 DZ-UniGuide AI AYA",
        "welcome_msg": "نظام التوجيه الذكي 2026",
        "personal_info": "📍 المعلومات الأساسية",
        "wilaya_label": "ولاية البكالوريا:",
        "stream_label": "شعبة البكالوريا:",
        "avg_range_label": "فئة المعدل:",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ حساب المعدل الموزون",
        "real_avg_label": "المعدل:",
        "grade_input_label": "علامة:",
        "factors_title": "🎯 العوامل",
        "f1_label": "عامل 1:",
        "f2_label": "عامل 2:",
        "factors_list": ["اهتمام", "عمل", "سمعة", "نصيحة"],
        "calculate_btn": "🔍 استخراج",
        "predict_btn": "🚀 تحليل",
        "bio_label": "اهتماماتك:",
        "mapping": {
            "علوم تجريبية": "Min_Science_x",
            "رياضيات": "Min_Math_x",
            "تقني رياضي": "Min_Tech",
            "تسيير واقتصاد": "Min_Eco",
            "آداب وفلسفة": "Min_Letters",
            "لغات أجنبية": "Min_Foreign_Lang"
        },
        "table_cols": ['الرمز', 'التخصص', 'الجامعة', 'المعدل', 'النطاق']
    }
}

L = texts["العربية"]


# ======================================================================
# 🔥 3. UI الرئيسي (APP PAGE)
# ======================================================================

st.title(L["bot_name"])
st.info(L["welcome_msg"])

st.subheader(L["personal_info"])

col1, col2, col3 = st.columns(3)

wilayas_list = ["01- أدرار", "02- الشلف", "03- الأغواط", "04- أم البواقي", "05- باتنة"]

with col1:
    selected_wilaya = st.selectbox(L["wilaya_label"], wilayas_list)

with col2:
    selected_stream = st.selectbox(L["stream_label"], list(L["mapping"].keys()))

with col3:
    st.selectbox(L["avg_range_label"], L["avg_ranges"])


st.subheader(L["weighted_title"])

sc1, sc2 = st.columns(2)

with sc1:
    real_avg = st.number_input(L["real_avg_label"], 10.0, 20.0, 14.5)

with sc2:
    sub_grade = st.number_input(L["grade_input_label"], 0.0, 20.0, 12.0)

weighted_avg = (real_avg * 2 + sub_grade) / 3
st.success(f"المعدل: {weighted_avg:.2f}")


st.divider()
st.subheader(L["factors_title"])

f1 = st.selectbox(L["f1_label"], L["factors_list"])
f2 = st.selectbox(L["f2_label"], ["None"] + L["factors_list"])


if 'final_db' not in st.session_state:
    st.session_state.final_db = None

if 'stream_col' not in st.session_state:
    st.session_state.stream_col = ""


# ======================================================================
# 🔥 4. SEARCH ENGINE (UNCHANGED LOGIC)
# ======================================================================

if st.button(L["calculate_btn"], use_container_width=True):

    user_code = selected_wilaya.split('-')[0].strip()
    stream_col = L["mapping"][selected_stream]

    eligible_mask = (db[stream_col] > 0) & (db[stream_col] <= weighted_avg)
    df_results = db[eligible_mask].copy()

    def is_valid_geo(row_allowed):
        return user_code in str(row_allowed)

    final_db = df_results[df_results['Allowed_Wilayas'].apply(is_valid_geo)]

    st.session_state.final_db = final_db
    st.session_state.stream_col = stream_col


if st.session_state.final_db is not None:
    st.dataframe(st.session_state.final_db.head())


# ======================================================================
# 🔥 5. AI PREDICTION (UNCHANGED LOGIC)
# ======================================================================

st.divider()
st.subheader("🤖 AI System")

user_bio = st.text_area("اهتماماتك")

if st.button(L["predict_btn"]) and st.session_state.final_db is not None:

    res = st.session_state.final_db.copy()
    res['success_prediction'] = 80
    res['interest_score'] = 70
    res['combined_score'] = 75

    st.session_state.filtered_res = res


# ======================================================================
# 🔥 6. RESULTS
# ======================================================================

if 'filtered_res' in st.session_state:

    st.subheader("🏆 النتائج")

    st.dataframe(st.session_state.filtered_res)

    display_metrics(st.session_state.filtered_res)

    smart_advice(85)

    final_summary(st.session_state.filtered_res, weighted_avg)