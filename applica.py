import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. تحميل البيانات والأصول ---
@st.cache_resource
def load_assets():
    try:
        db = pd.read_csv('MASTER_DATABASE_FINAL.csv')
        
        encoders = {}
        for filename in ['best_model.pkl', 'stream_encoder.pkl', 'factor_encoder.pkl', 'major_encoder.pkl']:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                encoders[filename] = obj
        
        model = encoders['best_model.pkl']
        stream_encoder = encoders['stream_encoder.pkl']
        factor_encoder = encoders['factor_encoder.pkl'] 
        major_encoder = encoders['major_encoder.pkl']
        
        return db, model, stream_encoder, factor_encoder, major_encoder
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

db, model, stream_encoder, factor_encoder, major_encoder = load_assets()

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="DZ-UniGuide AI", page_icon="🎓", layout="wide")

def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            color: black;
        }
        .stButton>button {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            border-radius: 10px;
            height: 3em;
            font-size: 18px;
        }
        .card {
            background-color: white;
            color: black;
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            text-align: center;
            margin: 10px 0;
        }
        .section-title {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

set_background()

# --- 3. تهيئة Session State ---
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if 'language' not in st.session_state:
    st.session_state.language = "العربية"
if 'show_second_section' not in st.session_state:
    st.session_state.show_second_section = False
if 'final_db' not in st.session_state:
    st.session_state.final_db = None
if 'filtered_res' not in st.session_state:
    st.session_state.filtered_res = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'stream_col' not in st.session_state:
    st.session_state.stream_col = None
if 'weighted_avg' not in st.session_state:
    st.session_state.weighted_avg = None
if 'min_required_col' not in st.session_state:
    st.session_state.min_required_col = None
if 'selected_stream' not in st.session_state:
    st.session_state.selected_stream = None
if 'selected_wilaya' not in st.session_state:
    st.session_state.selected_wilaya = None
if 'f1' not in st.session_state:
    st.session_state.f1 = None

# --- 4. قاموس الترجمة الكامل ---
texts = {
    "العربية": {
        "app_name": "🤖 DZ-UniGuide AI",
        "back_home": "🏠 الرئيسية",
        "settings": "الإعدادات",
        "language": "اللغة",
        "welcome_subtitle": "مساعدك الذكي لاختيار التخصص الجامعي 🎓",
        "start_btn": "🚀 ابدأ رحلتك الآن",
        "congrats_title": "مبروك نجاحك في البكالوريا",
        "congrats_text": "هذا إنجاز كبير في حياتك، وبداية مرحلة جديدة.",
        "advice_title": "نصيحة:",
        "advice_text": "اختيار التخصص أهم من المعدل، ركز على:",
        "advice_items": ["اهتماماتك", "مستقبلك المهني", "قدراتك الحقيقية"],
        "advice_footer": "نحن هنا لمساعدتك في اتخاذ القرار الصحيح",
        "bot_name": "🤖 DZ-UniGuide AI",
        "welcome_msg": "نظام المسح الشامل المعتمد على المنشور الوزاري وبيانات الاستبيان.",
        "personal_info": "📍 المعلومات الأساسية",
        "wilaya_label": "ولاية البكالوريا:",
        "stream_label": "شعبة البكالوريا:",
        "avg_range_label": "فئة المعدل:",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ حساب المعدل الموزون",
        "real_avg_label": "المعدل العام:",
        "grade_input_label": "علامة المادة الأساسية:",
        "math_grade": "علامة الرياضيات:",
        "tech_grade": "علامة التقنية:",
        "eco_grade": "علامة الاقتصاد:",
        "calc_method": "طريقة الحساب:",
        "factors_title": "🎯 العوامل المؤثرة",
        "f1_label": "العامل الأول:",
        "f2_label": "العامل الثاني:",
        "factors_list": ["الاهتمامات الشخصية", "فرص العمل", "سمعة الجامعة", "نصيحة الأهل"],
        "calculate_btn": "📜 استخراج القائمة",
        "next_btn": "➡️ التالي",
        "back_btn": "⬅️ رجوع",
        "predict_btn": "🧠 تحليل اهتماماتي",
        "bio_label": "✍️ صف اهتماماتك:",
        "bio_placeholder": "مثال: أحب البرمجة، الذكاء الاصطناعي، الاقتصاد، العلوم الطبيعية...",
        "streams": {
            "علوم تجريبية": "Min_Science_x",
            "رياضيات": "Min_Math_x", 
            "تقني رياضي": "Min_Tech",
            "تسيير واقتصاد": "Min_Eco",
            "آداب وفلسفة": "Min_Letters",
            "لغات أجنبية": "Min_Foreign_Lang"
        },
        "col_code": "الرمز",
        "col_major": "التخصص",
        "col_university": "الجامعة",
        "col_required": "المعدل المطلوب",
        "col_success": "نجاح %",
        "col_interest": "اهتمام %",
        "col_total": "الإجمالي %",
        "results_title": "🏆 نتائج التحليل",
        "best_10": "🏆 أفضل 10 تخصصات",
        "best_match": "⭐ أفضل تطابق",
        "success_rate": "💎 نسبة النجاح",
        "interest_match": "💖 توافق الاهتمامات",
        "options_count": "📈 عدد الخيارات",
        "smart_advice": "💡 نصيحة ذكية",
        "excellent": "🎉 ممتاز! لديك خيارات قوية جداً.",
        "good": "👍 جيد جداً! خيارات مناسبة.",
        "improve": "⚠️ تحسين مطلوب. جرب تعديل المعدل.",
        "found_results": "✅ تم العثور على {} تخصصاً متاحاً.",
        "no_results": "❌ لم يتم العثور على نتائج.",
        "analysis_done": "📊 تم تحليل {} كلمة",
        "click_predict": "👈 اضغط على 'تحليل اهتماماتي'",
        "click_next": "👈 اضغط على 'التالي' لمواصلة التحليل",
        "second_title": "🎯 تحليل الاهتمامات",
        "advice_text1": "اكتب نصاً مفصلاً عن اهتماماتك",
        "advice_text2": "اذكر المجالات التي تحبها",
        "advice_text3": "تحدث عن هواياتك",
        "prediction_title": "🤖 نظام التنبؤ المتقدم"
    },
    "English": {
        "app_name": "🤖 DZ-UniGuide AI",
        "back_home": "🏠 Home",
        "settings": "Settings",
        "language": "Language",
        "welcome_subtitle": "Your Smart Assistant for University Major Selection 🎓",
        "start_btn": "🚀 Start Your Journey",
        "congrats_title": "Congratulations on Your Baccalaureate!",
        "congrats_text": "This is a great achievement and the beginning of a new phase.",
        "advice_title": "Advice:",
        "advice_text": "Choosing the right major is more important than your grade, focus on:",
        "advice_items": ["Your interests", "Your career future", "Your real capabilities"],
        "advice_footer": "We are here to help you make the right decision",
        "bot_name": "🤖 DZ-UniGuide AI",
        "welcome_msg": "Comprehensive scan system based on official ministerial data.",
        "personal_info": "📍 Basic Information",
        "wilaya_label": "Baccalaureate Wilaya:",
        "stream_label": "Baccalaureate Stream:",
        "avg_range_label": "Average Category:",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ Weighted Average",
        "real_avg_label": "General Average:",
        "grade_input_label": "Core Subject Grade:",
        "math_grade": "Mathematics Grade:",
        "tech_grade": "Technology Grade:",
        "eco_grade": "Economics Grade:",
        "calc_method": "Calculation Method:",
        "factors_title": "🎯 Influencing Factors",
        "f1_label": "1st Factor:",
        "f2_label": "2nd Factor:",
        "factors_list": ["Personal Interests", "Job Opportunities", "University Reputation", "Family Advice"],
        "calculate_btn": "📜 Extract List",
        "next_btn": "➡️ Next",
        "back_btn": "⬅️ Back",
        "predict_btn": "🧠 Analyze Interests",
        "bio_label": "✍️ Describe your interests:",
        "bio_placeholder": "Example: I love programming, AI, economics, natural sciences...",
        "streams": {
            "Experimental Sciences": "Min_Science_x",
            "Mathematics": "Min_Math_x",
            "Technical Math": "Min_Tech", 
            "Management & Economics": "Min_Eco",
            "Arts & Philosophy": "Min_Letters",
            "Foreign Languages": "Min_Foreign_Lang"
        },
        "col_code": "Code",
        "col_major": "Major",
        "col_university": "University",
        "col_required": "Required Grade",
        "col_success": "Success %",
        "col_interest": "Interest %",
        "col_total": "Total %",
        "results_title": "🏆 Analysis Results",
        "best_10": "🏆 Top 10 Majors",
        "best_match": "⭐ Best Match",
        "success_rate": "💎 Success Rate",
        "interest_match": "💖 Interest Match",
        "options_count": "📈 Options Count",
        "smart_advice": "💡 Smart Advice",
        "excellent": "🎉 Excellent! You have very strong options.",
        "good": "👍 Very good! Suitable options available.",
        "improve": "⚠️ Improvement needed. Try adjusting your average.",
        "found_results": "✅ Found {} available majors.",
        "no_results": "❌ No results found.",
        "analysis_done": "📊 Analyzed {} keywords",
        "click_predict": "👈 Click 'Analyze Interests'",
        "click_next": "👈 Click 'Next' to continue",
        "second_title": "🎯 Interest Analysis",
        "advice_text1": "Write a detailed text about your interests",
        "advice_text2": "Mention the fields you like",
        "advice_text3": "Talk about your hobbies",
        "prediction_title": "🤖 Advanced Prediction System"
    }
}

# --- 5. السايدبار للغة ---
with st.sidebar:
    st.title(texts[st.session_state.language]["settings"])
    selected_lang = st.selectbox(
        texts[st.session_state.language]["language"],
        ["العربية", "English"],
        index=0 if st.session_state.language == "العربية" else 1
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

L = texts[st.session_state.language]

# --- 6. صفحة الترحيب ---
if st.session_state.page == "welcome":
    st.markdown(f"""
    <div style="text-align:center">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" width="120">
        <h1 style="color:#00e5ff;">{L["app_name"]}</h1>
        <p style="font-size:20px;">{L["welcome_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h3>{L["start_btn"]}</h3>
        <p>{L["congrats_text"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div align="center">
        <h2>{L["congrats_title"]}</h2>
        <p>{L["congrats_text"]}</p>
        <p><b>{L["advice_title"]}</b> {L["advice_text"]}</p>
        <p>• {L["advice_items"][0]}</p>
        <p>• {L["advice_items"][1]}</p>
        <p>• {L["advice_items"][2]}</p>
        <p>{L["advice_footer"]}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(L["start_btn"], use_container_width=True):
        st.session_state.page = "app"
        st.rerun()
    st.stop()

# --- 7. التطبيق الرئيسي ---
if st.session_state.page == "app":
    
    col_back1, col_back2 = st.columns([1, 5])
    with col_back1:
        if st.button(L["back_home"]):
            st.session_state.page = "welcome"
            st.session_state.show_second_section = False
            st.rerun()
    
    st.markdown(f"""
    <div style="text-align:center">
        <h1 style="color:#00e5ff;">{L["bot_name"]}</h1>
        <p style="font-size:18px;">{L["welcome_msg"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== القسم الأول ==========
    if not st.session_state.show_second_section:
        
        st.markdown(f'<div class="card"><h3>{L["personal_info"]}</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        wilayas = ["01- أدرار", "02- الشلف", "03- الأغواط", "04- أم البواقي", "05- باتنة", "06- بجاية",
                   "07- بسكرة", "08- بشار", "09- البليدة", "10- البويرة", "11- تمنراست", "12- تبسة",
                   "13- تلمسان", "14- تيارت", "15- تيزي وزو", "16- الجزائر", "17- الجلفة", "18- جيجل",
                   "19- سطيف", "20- سعيدة", "21- سكيكدة", "22- سيدي بلعباس", "23- عنابة", "24- قالمة",
                   "25- قسنطينة", "26- المدية", "27- مستغانم", "28- المسيلة", "29- معسكر", "30- ورقلة",
                   "31- وهران", "32- البيض", "33- إليزي", "34- برج بوعريريج", "35- بومرداس", "36- الطارف",
                   "37- تندوف", "38- تسمسيلت", "39- الوادي", "40- خنشلة", "41- سوق أهراس", "42- تيبازة",
                   "43- ميلة", "44- عين الدفلى", "45- النعامة", "46- عين تموشنت", "47- غرداية", "48- غليزان",
                   "49- تيميمون", "50- برج باجي مختار", "51- أولاد جلال", "52- بني عباس", "53- عين صالح",
                   "54- عين قزام", "55- تقرت", "56- جانت", "57- المغير", "58- المنيعة"]
        
        with col1:
            selected_wilaya = st.selectbox(L["wilaya_label"], wilayas)
        with col2:
            stream_options = list(L["streams"].keys())
            selected_stream = st.selectbox(L["stream_label"], stream_options)
        with col3:
            avg_range = st.selectbox(L["avg_range_label"], L["avg_ranges"])
        
        st.subheader(L["weighted_title"])
        col1, col2 = st.columns(2)
        with col1:
            real_avg = st.number_input(L["real_avg_label"], 10.0, 20.0, 12.0, 0.01)
        with col2:
            if "علوم" in selected_stream or "رياضيات" in selected_stream or "Experimental" in selected_stream or "Mathematics" in selected_stream:
                sub_grade = st.number_input(L["grade_input_label"], 0.0, 20.0, 10.0, 0.01)
            elif "تقني" in selected_stream or "Technical" in selected_stream:
                sub_grade = st.number_input(L["tech_grade"], 0.0, 20.0, 10.0, 0.01)
            elif "تسيير" in selected_stream or "Management" in selected_stream:
                sub_grade = st.number_input(L["eco_grade"], 0.0, 20.0, 10.0, 0.01)
            else:
                sub_grade = st.number_input(L["grade_input_label"], 0.0, 20.0, 10.0, 0.01)
        
        if "تقني" in selected_stream or "Technical" in selected_stream:
            extra_grade = st.number_input(L["math_grade"], 0.0, 20.0, 10.0, 0.01)
            weighted_avg = (real_avg * 2 + (sub_grade + extra_grade) / 2) / 3
        else:
            weighted_avg = (real_avg * 2 + sub_grade) / 3
        
        st.info(f"{L['calc_method']} {weighted_avg:.2f}")
        
        st.divider()
        st.subheader(L["factors_title"])
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.selectbox(L["f1_label"], L["factors_list"])
        with col2:
            f2_options = ["None"] + [x for x in L["factors_list"] if x != f1]
            f2 = st.selectbox(L["f2_label"], f2_options)
        
        if st.button(L["calculate_btn"], use_container_width=True):
            user_code = selected_wilaya.split('-')[0].strip()
            user_name = selected_wilaya.split('-')[1].strip()
            stream_col = L["streams"][selected_stream]
            
            eligible = (db[stream_col] > 0) & (db[stream_col] <= weighted_avg)
            results = db[eligible].copy()
            
            def valid_geo(row):
                allowed = str(row).lower()
                return (user_code in allowed.split()) or ('national' in allowed)
            
            results = results[results['Allowed_Wilayas'].apply(valid_geo)].copy()
            
            if not results.empty:
                def set_rank(row):
                    if user_name.lower() in str(row['University_Name']).lower():
                        return 3
                    return 1
                
                results['rank'] = results.apply(set_rank, axis=1)
                results = results.sort_values(['rank', stream_col], ascending=[False, False])
                
                st.session_state.final_db = results
                st.session_state.stream_col = stream_col
                st.session_state.weighted_avg = weighted_avg
                st.session_state.selected_stream = selected_stream
                st.session_state.selected_wilaya = selected_wilaya
                st.session_state.f1 = f1
                
                st.success(L["found_results"].format(len(results)))
                st.balloons()
            else:
                st.error(L["no_results"])
        
        # الجدول الأول
        if st.session_state.final_db is not None:
            df = st.session_state.final_db[['Code_Fil', 'University_Major', 'University_Name', 
                                             st.session_state.stream_col, 'Allowed_Wilayas']].copy()
            df.columns = [L["col_code"], L["col_major"], L["col_university"], L["col_required"], "Scope"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(L["next_btn"], use_container_width=True):
                    st.session_state.show_second_section = True
                    st.rerun()
    
    # ========== القسم الثاني: الاهتمامات والتحليل ==========
    else:
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button(L["back_btn"]):
                st.session_state.show_second_section = False
                st.rerun()
        
        st.markdown(f'<div class="section-title">{L["second_title"]}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <p>💡 <b>{L["advice_title"]}</b></p>
            <p>✓ {L["advice_text1"]}</p>
            <p>✓ {L["advice_text2"]}</p>
            <p>✓ {L["advice_text3"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(L["prediction_title"])
        
        user_bio = st.text_area(L["bio_label"], placeholder=L["bio_placeholder"], height=150)
        
        if st.button(L["predict_btn"], use_container_width=True):
            with st.spinner("🧠 Processing..."):
                res = st.session_state.final_db.copy()
                w_avg = st.session_state.weighted_avg
                s_stream = st.session_state.selected_stream
                fact = st.session_state.f1
                
                stream_enc = 0
                factor_enc = 0
                try:
                    stream_enc = int(stream_encoder.transform([[s_stream]])[0])
                    factor_enc = int(factor_encoder.transform([[fact]])[0])
                except:
                    pass
                
                major_enc = {}
                for m in res['University_Major'].unique():
                    m_str = str(m).lower()
                    if any(w in m_str for w in ['science', 'علوم', 'tech', 'هندسة']):
                        major_enc[m] = 0
                    elif any(w in m_str for w in ['health', 'طب', 'medicine']):
                        major_enc[m] = 1
                    elif any(w in m_str for w in ['eco', 'اقتصاد', 'management']):
                        major_enc[m] = 2
                    else:
                        major_enc[m] = 0
                
                def predict(row):
                    features = np.array([[stream_enc, w_avg, factor_enc, major_enc.get(row['University_Major'], 0)]])
                    try:
                        return round(model.predict_proba(features)[0][1] * 100, 2)
                    except:
                        diff = w_avg - row[st.session_state.stream_col]
                        return round(min(max(65 + diff * 5, 40), 98), 2)
                
                res['success_prediction'] = res.apply(predict, axis=1)
                
                if user_bio and user_bio.strip():
                    try:
                        def clean(t):
                            t = str(t).lower()
                            t = re.sub(r'[ًٌٍَُِّْ]', '', t)
                            t = re.sub(r'[^\w\s]', ' ', t)
                            return re.sub(r'\s+', ' ', t).strip()
                        
                        bio = clean(user_bio)
                        mapping = {
                            'برمجة': 'programming computer', 'بيانات': 'data analytics',
                            'رياضيات': 'mathematics', 'فيزياء': 'physics',
                            'علوم': 'science', 'هندسة': 'engineering',
                            'طب': 'medicine', 'اقتصاد': 'economics',
                            'تسيير': 'management', 'قانون': 'law',
                            'لغات': 'languages', 'فن': 'art'
                        }
                        
                        enhanced = bio
                        for ar, en in mapping.items():
                            if ar in bio:
                                enhanced += ' ' + en
                        
                        words = bio.split()
                        majors_txt = [clean(m) for m in res['University_Major']]
                        
                        docs = [enhanced] + majors_txt
                        tfidf = TfidfVectorizer(max_features=500)
                        matrix = tfidf.fit_transform(docs)
                        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
                        
                        scores = [min(55 + s * 200, 98) for s in sims]
                        for i, m in enumerate(res['University_Major']):
                            bonus = sum(5 for w in words if len(w) > 2 and w in str(m).lower())
                            scores[i] = min(scores[i] + bonus, 98)
                        
                        res['interest_score'] = [round(s, 1) for s in scores]
                        st.info(L["analysis_done"].format(len(words)))
                    except Exception as e:
                        res['interest_score'] = 60
                else:
                    res['interest_score'] = 60
                
                res['combined_score'] = res['success_prediction'] * 0.7 + res['interest_score'] * 0.3
                all_res = res.sort_values('combined_score', ascending=False).drop_duplicates('University_Major')
                
                col = st.session_state.stream_col
                filtered = all_res[(all_res[col] <= w_avg) & (all_res[col] >= w_avg - 3) & (all_res['success_prediction'] > 70)].copy()
                
                st.session_state.filtered_res = filtered
                st.session_state.analysis_done = True
                st.session_state.min_required_col = col
        
        # الجدول الثاني (النتائج الكاملة مع الألوان)
        if st.session_state.get('analysis_done') and st.session_state.filtered_res is not None:
            filtered = st.session_state.filtered_res
            w_avg = st.session_state.weighted_avg
            col = st.session_state.min_required_col
            
            if not filtered.empty:
                st.success(f"🏆 {L['results_title']} - {L['col_required']}: {w_avg:.1f}")
                
                display_df = filtered.sort_values([col, 'combined_score'], ascending=[False, False])
                display_cols = ['Code_Fil', 'University_Major', 'University_Name', col, 
                               'success_prediction', 'interest_score', 'combined_score']
                display_df = display_df[display_cols].copy()
                display_df.columns = [L["col_code"], L["col_major"], L["col_university"], 
                                     L["col_required"], L["col_success"], L["col_interest"], L["col_total"]]
                
                def color_success(val):
                    if isinstance(val, (int, float)):
                        if val > 85: return 'background-color: #4caf50; color: white'
                        elif val > 75: return 'background-color: #ffca28; color: black'
                        else: return 'background-color: #f44336; color: white'
                    return ''
                
                styled_df = display_df.style \
                    .map(lambda x: 'background-color: #4caf50; color: white' if isinstance(x, (int, float)) and x <= w_avg else '', subset=[L["col_required"]]) \
                    .map(color_success, subset=[L["col_success"]]) \
                    .background_gradient(cmap='RdYlGn', subset=[L["col_total"]]) \
                    .format("{:.1f}%", subset=[L["col_success"], L["col_interest"], L["col_total"]]) \
                    .format("{:.2f}", subset=[L["col_required"]])
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # الجدول الثالث (أفضل 10 تخصصات)
                st.markdown("---")
                st.subheader(L["best_10"])
                
                top_10 = filtered.sort_values(
                    by=['combined_score', 'interest_score', 'success_prediction'],
                    ascending=[False, False, False]
                ).head(10).copy()
                
                top_10_display = top_10[['University_Major', 'University_Name', col, 
                                        'success_prediction', 'interest_score', 'combined_score']].copy()
                top_10_display.columns = [L["col_major"], L["col_university"], L["col_required"], 
                                         L["col_success"], L["col_interest"], L["col_total"]]
                top_10_display = top_10_display.round(1)
                
                def color_required(val):
                    if isinstance(val, (int, float)):
                        if val >= w_avg: return 'background-color: #d32f2f; color: white'
                        elif val >= w_avg - 1: return 'background-color: #f57c00; color: white'
                        elif val >= w_avg - 2: return 'background-color: #ffca28'
                        else: return 'background-color: #4caf50; color: white'
                    return ''
                
                def color_interest(val):
                    if isinstance(val, (int, float)):
                        if val > 85: return 'background-color: #e91e63; color: white'
                        elif val > 75: return 'background-color: #9c27b0; color: white'
                        elif val > 65: return 'background-color: #673ab7; color: white'
                        else: return 'background-color: #607d8b; color: white'
                    return ''
                
                def color_success_top(val):
                    if isinstance(val, (int, float)):
                        if val > 85: return 'background-color: #4caf50; color: white'
                        elif val > 75: return 'background-color: #8bc34a'
                        elif val > 65: return 'background-color: #ffc107'
                        else: return 'background-color: #f44336; color: white'
                    return ''
                
                styled_top10 = top_10_display.style \
                    .map(color_required, subset=[L["col_required"]]) \
                    .map(color_interest, subset=[L["col_interest"]]) \
                    .map(color_success_top, subset=[L["col_success"]]) \
                    .background_gradient(cmap='RdYlGn', subset=[L["col_total"]]) \
                    .format("{:.1f}%", subset=[L["col_success"], L["col_interest"], L["col_total"]]) \
                    .format("{:.2f}", subset=[L["col_required"]])
                
                st.dataframe(styled_top10, use_container_width=True, height=350)
                
                # ملخص البطاقات
                st.markdown("---")
                st.subheader("🎉 ملخص تحليلك الشخصي")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(L["best_match"], f"{filtered['combined_score'].max():.1f}%")
                m2.metric(L["success_rate"], f"{filtered['success_prediction'].max():.1f}%")
                m3.metric(L["interest_match"], f"{filtered['interest_score'].max():.1f}%")
                m4.metric(L["options_count"], len(filtered))
                
                # نصيحة ذكية
                st.markdown("---")
                st.subheader(L["smart_advice"])
                best_match = filtered['combined_score'].max()
                if best_match > 85:
                    st.success(L["excellent"])
                elif best_match > 70:
                    st.info(L["good"])
                else:
                    st.warning(L["improve"])
                
                avg_required = filtered[col].mean()
                st.info(f"📏 {L['col_required']}: {avg_required:.2f} | {L['real_avg_label']}: {w_avg:.2f}")
                
                total_specialties = len(filtered)
                best_match_pct = f"{filtered['combined_score'].max():.1f}%"
                
                st.markdown(f"""
                    <div style='text-align: center; background-color: #E8F5E8; padding: 25px; border-radius: 15px; 
                            border-left: 6px solid #4CAF50; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                        <h2 style='color: #2E7D32; margin: 0 0 15px 0;'>✅ {L['results_title']}</h2>
                        <div style='font-size: 18px; line-height: 1.6;'>
                            • <b style='color: #1976D2;'>{total_specialties}</b> {L['options_count']}<br>
                            • {L['best_match']}: <b style='color: #E91E63;'>{best_match_pct}</b><br>
                            • <b style='color: #4CAF50;'>{L['smart_advice']}</b>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(L["no_results"])
        else:
            st.info(L["click_predict"])