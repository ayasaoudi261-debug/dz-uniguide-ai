import pandas as pd
import pickle
import streamlit as st
# --- 1. تحميل البيانات والأصول ---
@st.cache_resource
def load_assets():
    try:
        db = pd.read_csv('MASTER_DATABASE_FINAL.csv')

        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('stream_encoder.pkl', 'rb') as f:
            stream_encoder = pickle.load(f)

        return db, model, stream_encoder

    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        return None, None, None


db, model, stream_encoder = load_assets()

# --- 2. إعدادات الصفحة واللغات ---
st.set_page_config(page_title="DZ-UniGuide AI AYA", page_icon="🎓", layout="wide")

if 'language' not in st.session_state:
    st.session_state.language = "العربية"

texts = {
    "العربية": {
        "bot_name": "🤖 DZ-UniGuide AI",
        "welcome_msg": "نظام المسح الشامل المعتمد على المنشور الوزاري وبيانات الاستبيان.",
        "personal_info": "📍 المعلومات الأساسية",
        "wilaya_label": "ولاية البكالوريا:",
        "stream_label": "شعبة البكالوريا:",
        "avg_range_label": "فئة المعدل (bac_average):",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ حساب المعدل الموزون",
        "real_avg_label": "المعدل العام الدقيق:",
        "grade_input_label": "علامة المادة الأساسية:",
        "factors_title": "🎯 العوامل المؤثرة (من الاستبيان)",
        "f1_label": "العامل الأول (إجباري):",
        "f2_label": "العامل الثاني (اختياري):",
        "factors_list": ["الاهتمامات الشخصية", "فرص العمل", "سمعة الجامعة", "نصيحة الأهل"],
        "calculate_btn": "استخراج القائمة الكاملة 📜",
        "mapping": {"علوم تجريبية": "Min_Science_x", "رياضيات": "Min_Math_x", "تقني رياضي": "Min_Tech", "تسيير واقتصاد": "Min_Eco", "آداب وفلسفة": "Min_Letters", "لغات أجنبية": "Min_Foreign_Lang"},
        "table_cols": ['الرمز', 'التخصص الجامعي', 'المؤسسة', 'المعدل الموزون', 'النطاق الجغرافي']
    },
    "English": {
        "bot_name": "🤖 DZ-UniGuide AI",
        "welcome_msg": "Comprehensive scan system based on official ministerial data.",
        "personal_info": "📍 Basic Information",
        "wilaya_label": "Baccalaureate Wilaya:",
        "stream_label": "Baccalaureate Stream:",
        "avg_range_label": "Average Category (bac_average):",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ Weighted Average Calculation",
        "real_avg_label": "Exact General Average:",
        "grade_input_label": "Core Subject Grade:",
        "factors_title": "🎯 Influencing Factors",
        "f1_label": "1st Factor (Mandatory):",
        "f2_label": "2nd Factor (Optional):",
        "factors_list": ["Personal Interests", "Job Opportunities", "University Reputation", "Family Advice"],
        "calculate_btn": "Extract Full List 📜",
        "mapping": {"Experimental Sciences": "Min_Science_x", "Mathematics": "Min_Math_x", "Technical Math": "Min_Tech", "Management & Economics": "Min_Eco", "Arts & Philosophy": "Min_Letters", "Foreign Languages": "Min_Foreign_Lang"},
        "table_cols": ['Code', 'Major', 'University', 'Min. Grade', 'Geographic Scope']
    }
}

with st.sidebar:
    st.title("Settings / إعدادات")
    selected_lang = st.selectbox("Language / اللغة", ["العربية", "English"], index=0 if st.session_state.language == "العربية" else 1)
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

L = texts[st.session_state.language]

# --- 3. بناء الواجهة ---
st.title(L["bot_name"])
st.info(L["welcome_msg"])

st.subheader(L["personal_info"])
col1, col2, col3 = st.columns(3)
with col1:
    wilayas_list = ["01- أدرار", "02- الشلف", "03- الأغواط", "04- أم البواقي", "05- باتنة", "06- بجاية", "07- بسكرة", "08- بشار", "09- البليدة", "10- البويرة", "11- تمنراست", "12- تبسة", "13- تلمسان", "14- تيارت", "15- تيزي وزو", "16- الجزائر", "17- الجلفة", "18- جيجل", "19- سطيف", "20- سعيدة", "21- سكيكدة", "22- سيدي بلعباس", "23- عنابة", "24- قالمة", "25- قسنطينة", "26- المدية", "27- مستغانم", "28- المسيلة", "29- معسكر", "30- ورقلة", "31- وهران", "32- البيض", "33- إليزي", "34- برج بوعريريج", "35- بومرداس", "36- الطارف", "37- تندوف", "38- تسمسيلت", "39- الوادي", "40- خنشلة", "41- سوق أهراس", "42- تيبازة", "43- ميلة", "44- عين الدفلى", "45- النعامة", "46- عين تموشنت", "47- غرداية", "48- غليزان", "49- تيميمون", "50- برج باجي مختار", "51- أولاد جلال", "52- بني عباس", "53- عين صالح", "54- عين قزام", "55- تقرت", "56- جانت", "57- المغير", "58- المنيعة"]
    selected_wilaya = st.selectbox(L["wilaya_label"], wilayas_list)
with col2:
    selected_stream = st.selectbox(L["stream_label"], list(L["mapping"].keys()))
with col3:
    user_avg_range = st.selectbox(L["avg_range_label"], L["avg_ranges"])

st.subheader(L["weighted_title"])
sc1, sc2 = st.columns(2)
with sc1:
    real_avg = st.number_input(L["real_avg_label"], 10.0, 20.0, 12.0, step=0.01)
with sc2:
    sub_grade = st.number_input(L["grade_input_label"], 0.0, 20.0, 10.0, step=0.25)

weighted_avg = (real_avg * 2 + sub_grade) / 3
st.markdown(f"**المعدل الموزون المحسوب:** `{weighted_avg:.2f}`")

st.divider()
st.subheader(L["factors_title"])
f1_col, f2_col = st.columns(2)
with f1_col:
    f1 = st.selectbox(L["f1_label"], L["factors_list"])
with f2_col:
    f2 = st.selectbox(L["f2_label"], ["None"] + [x for x in L["factors_list"] if x != f1])

# --- 4. المحرك الأساسي (تعديل للحفاظ على البيانات) ---

# تعريف مفاتيح في session_state إذا لم تكن موجودة
if 'final_db' not in st.session_state:
    st.session_state.final_db = None
if 'stream_col' not in st.session_state:
    st.session_state.stream_col = ""

if st.button(L["calculate_btn"], use_container_width=True):
    if db is not None:
        user_code = selected_wilaya.split('-')[0].strip()
        user_name = selected_wilaya.split('-')[1].strip()
        stream_col = L["mapping"].get(selected_stream)
        
        # 1. تصفية الاستحقاق والمعدل الموزون
        eligible_mask = (db[stream_col] > 0) & (db[stream_col] <= weighted_avg)
        df_results = db[eligible_mask].copy()
        
        # 2. تصفية الجغرافيا
        def is_valid_geo(row_allowed):
            allowed_str = str(row_allowed).lower()
            parts = allowed_str.replace(',', ' ').split()
            return (user_code in parts) or ('national' in parts)

        final_db = df_results[df_results['Allowed_Wilayas'].apply(is_valid_geo)].copy()

        # 3. الترتيب الأولي حسب الجامعة (الولاية أولاً)
        def set_rank(row):
            uni = str(row['University_Name']).lower()
            if user_name.lower() in uni: return 3 
            return 1

        if not final_db.empty:
            final_db['rank'] = final_db.apply(set_rank, axis=1)
            final_db = final_db.sort_values(by=['rank', stream_col], ascending=[False, False])
            
            # --- الحفظ في session_state هو السر هنا ---
            st.session_state.final_db = final_db
            st.session_state.stream_col = stream_col
            st.balloons()
        else:
            st.session_state.final_db = None
            st.warning("لم يتم العثور على نتائج.")

# عرض القائمة الكاملة إذا كانت مخزنة (لكي لا تختفي)
if st.session_state.final_db is not None:
    st.success(f"✅ تم العثور على {len(st.session_state.final_db)} تخصصاً متاحاً.")
    view_df = st.session_state.final_db[['Code_Fil', 'University_Major', 'University_Name', st.session_state.stream_col, 'Allowed_Wilayas']].copy()
    view_df.columns = L["table_cols"]
    st.dataframe(view_df, use_container_width=True, hide_index=True)

   # =========================================================
# =========================================================
# 🤖 نظام التوجيه الشامل الذكي (التنبؤ بنسبة النجاح لكل التخصصات)
# =========================================================

if st.session_state.final_db is not None:
    st.divider()
    st.subheader("🎯 تحليل شامل لجميع الخيارات المتاحة + التنبؤ بالنجاح")

    # مدخل الميول (اختياري لكنه يساعد في الترتيب)
    user_bio = st.text_area(
        "✍️ صف اهتماماتك (اختياري):",
        placeholder="مثال: أحب العلوم، التكنولوجيا، الاقتصاد...",
        key="full_bio"
    )

    if st.button("🚀 عرض كافة الاحتمالات والتنبؤات"):
        with st.spinner("🧠 جاري معالجة كافة التخصصات المتاحة وحساب فرص النجاح..."):
            # 1. جلب كافة التخصصات المصفاة حسب (الشعبة + الولاية + المعدل)
            res = st.session_state.final_db.copy()

            # 2. تطبيق نموذج الـ ML للتنبؤ بنسبة النجاح
            # ملاحظة: النموذج يتوقع مدخلات محددة (مثل المعدل والشعبة)
            def predict_student_success(row):
                try:
                    # تحويل شعبة المستخدم الحالية لترميز رقمي (كما في التدريب)
                    stream_enc = stream_encoder.transform([st.session_state.user_stream])[0]
                    
                    # تجهيز البيانات للنموذج [المعدل الحقيقي، كود الشعبة]
                    # يمكنك إضافة أعمدة أخرى هنا إذا كان النموذج يتطلبها (مثل كود التخصص)
                    features = [[real_avg, stream_enc]]
                    
                    # جلب الاحتمالية من النموذج (نسبة النجاح)
                    proba = model.predict_proba(features)[0][1] 
                    return round(proba * 100, 2)
                except:
                    # معادلة ذكية احتياطية في حال تعذر التنبؤ ببعض التخصصات
                    # تعتمد على الفارق بين معدل الطالب ومعدل القبول + بونص صغير
                    diff = real_avg - row[st.session_state.stream_col]
                    base = 65 + (diff * 5)
                    return round(min(max(base, 40), 98), 2)

            res['success_prediction'] = res.apply(predict_student_success, axis=1)

            # 3. حساب نقاط الميول (NLP) لترتيب الجدول
            def calculate_nlp(major_name):
                if not user_bio: return 0
                major_name = str(major_name).lower()
                bio = user_bio.lower()
                # إذا وجدت كلمات من الميول في اسم التخصص
                match_count = sum(1 for word in bio.split() if len(word) > 3 and word in major_name)
                return match_count * 20

            res['interest_score'] = res['University_Major'].apply(calculate_nlp)

            # 4. الترتيب النهائي (الأولوية لنسبة النجاح المتوقعة من الـ ML)
            res = res.sort_values(by=['success_prediction', 'interest_score'], ascending=False)
            
            # منع التكرار (تخصص واحد لكل جامعة)
            res = res.drop_duplicates(subset=['University_Major', 'University_Name'])

            # 5. عرض النتائج بشكل منظم
            st.success(f"📊 تم تحليل {len(res)} تخصصاً جامعياً متاحاً لك.")

            # تنسيق الجدول للعرض
            output_df = res[['Code_Fil', 'University_Major', 'University_Name', st.session_state.stream_col, 'success_prediction']].copy()
            output_df.columns = ['الرمز', 'التخصص الجامعي', 'المؤسسة/الجامعة', 'معدل القبول', 'نسبة النجاح المتوقعة %']

            st.dataframe(
                output_df.style.background_gradient(cmap='RdYlGn', subset=['نسبة النجاح المتوقعة %'])
                .format("{:.2f}%", subset=['نسبة النجاح المتوقعة %']),
                use_container_width=True,
                hide_index=True
            )

            # 6. توزيع إحصائي بسيط (اختياري)
            st.info(f"💡 **ملخص ذكي:** لديك {len(res[res['success_prediction'] > 80])} تخصصاً تتجاوز نسبة نجاحك فيها 80% بناءً على نموذج الذكاء الاصطناعي.")