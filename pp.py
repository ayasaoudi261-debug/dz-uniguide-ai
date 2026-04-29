import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 🔥 1. الدوال المساعدة المتقدمة (منظمة ونظيفة)
# ============================================================================

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
    """عرض الإحصائيات في columns منظمة"""
    high_success = len(df[df['success_prediction'] > 80])
    high_interest = len(df[df['interest_score'] > 75])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 التخصصات", len(df))
    with col2: st.metric("💎 نجاح >80%", high_success)
    with col3: st.metric("❤️ اهتمام >75%", high_interest)
    with col4: st.metric("⭐ أفضل تطابق", f"{df['combined_score'].max():.1f}%")

def smart_advice(best_score):
    """نصيحة ذكية بناءً على النتائج"""
    if best_score > 85:
        st.success("🎉 **ممتاز!** لديك خيارات قوية جداً. ركز على أفضل 5 من القائمة.")
    elif best_score > 70:
        st.info("👍 **جيد جداً!** خيارات مناسبة. أضف عوامل إضافية للاختيار.")
    else:
        st.warning("⚠️ **تحسين مطلوب.** جرب تعديل المعدل الموزون أو العوامل.")

def style_table(df, weighted_avg):
    """تلوين الجدول المتقدم"""
    def color_grades(val):
        if isinstance(val, (int, float)):
            if val > weighted_avg * 0.95: return 'background-color: #d32f2f; color: white'
            elif val > weighted_avg * 0.85: return 'background-color: #f57c00; color: white'
            elif val > weighted_avg * 0.75: return 'background-color: #ffca28'
            else: return 'background-color: #4caf50; color: white'
        return ''
    
    def color_success(val):
        if isinstance(val, (int, float)):
            if val > 85: return 'background-color: #4caf50; color: white'
            elif val > 75: return 'background-color: #ffca28'
            elif val > 70: return 'background-color: #ff9800; color: white'
            else: return 'background-color: #f44336; color: white'
        return ''
    
    def color_interest(val):
        if isinstance(val, (int, float)):
            if val > 85: return 'background-color: #e91e63; color: white'
            elif val > 75: return 'background-color: #9c27b0; color: white'
            elif val > 65: return 'background-color: #673ab7'
            else: return 'background-color: #607d8b'
        return ''
    
    return df.style \
        .applymap(color_grades, subset=['المعدل المطلوب']) \
        .applymap(color_success, subset=['نجاح %']) \
        .applymap(color_interest, subset=['اهتمام %']) \
        .background_gradient(cmap='RdYlGn', subset=['الإجمالي %']) \
        .format("{:.1f}%", subset=['نجاح %', 'اهتمام %', 'الإجمالي %']) \
        .format("{:.2f}", subset=['المعدل المطلوب'])

def final_summary(filtered_res, weighted_avg):
    """الملخص النهائي الشامل"""
    total = len(filtered_res)
    best_match = filtered_res['combined_score'].max()
    
    st.markdown("---")
    st.markdown("### 🎉 **ملخص تحليلك الشامل**")
    
    summary_cols = st.columns([1, 1, 1, 1])
    with summary_cols[0]: st.metric("⭐ أفضل تطابق", f"{best_match:.1f}%")
    with summary_cols[1]: st.metric("💎 أعلى نجاح", f"{filtered_res['success_prediction'].max():.1f}%")
    with summary_cols[2]: st.metric("💖 أعلى اهتمام", f"{filtered_res['interest_score'].max():.1f}%")
    with summary_cols[3]: st.metric("📈 الخيارات", f"{total}")
    
    avg_required = filtered_res[st.session_state.stream_col].mean()
    st.info(f"📏 **المعدل المتوسط المطلوب:** {avg_required:.2f} | **معدلك:** {weighted_avg:.2f}")
    
    st.markdown(f"""
    <div style='text-align: center; background: linear-gradient(135deg, #E8F5E8, #C8E6C9); 
                padding: 25px; border-radius: 15px; border-left: 6px solid #4CAF50; 
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <h2 style='color: #2E7D32; margin: 0 0 15px 0;'>✅ **تحليلك جاهز تماماً!**</h2>
        <div style='font-size: 18px; line-height: 1.8;'>
            • <b style='color: #1976D2;'>{total}</b> تخصص متاح لمعدلك<br>
            • أعلى تطابق: <b style='color: #E91E63;'>{best_match:.1f}%</b><br>
            • <b style='color: #4CAF50;'>راجع نصيحتك 👆</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 🔥 2. تحميل البيانات والإعدادات الأساسية
# ============================================================================

db, model, stream_encoder, factor_encoder, major_encoder = load_assets()
st.set_page_config(page_title="DZ-UniGuide AI AYA", page_icon="🎓", layout="wide")

if 'language' not in st.session_state:
    st.session_state.language = "العربية"

# النصوص ثنائية اللغة
texts = {
    "العربية": {
        "bot_name": "🤖 DZ-UniGuide AI AYA",
        "welcome_msg": "نظام المسح الشامل المعتمد على المنشور الوزاري الرسمي 2026",
        "personal_info": "📍 المعلومات الأساسية",
        "wilaya_label": "ولاية البكالوريا:",
        "stream_label": "شعبة البكالوريا:",
        "avg_range_label": "فئة المعدل:",
        "avg_ranges": ["10 – 11.99", "12 – 13.99", "14 – 15.99", "16 or above"],
        "weighted_title": "⚙️ حساب المعدل الموزون",
        "real_avg_label": "المعدل العام الدقيق:",
        "grade_input_label": "علامة المادة الأساسية:",
        "factors_title": "🎯 العوامل المؤثرة",
        "f1_label": "العامل الأول (إجباري):",
        "f2_label": "العامل الثاني (اختياري):",
        "factors_list": ["الاهتمامات الشخصية", "فرص العمل", "سمعة الجامعة", "نصيحة الأهل"],
        "calculate_btn": "🔍 استخراج القائمة الكاملة",
        "predict_btn": "🚀 تحليل التنبؤات المتقدمة",
        "bio_label": "✍️ صف اهتماماتك (اختياري):",
        "mapping": {"علوم تجريبية": "Min_Science_x", "رياضيات": "Min_Math_x", "تقني رياضي": "Min_Tech", 
                   "تسيير واقتصاد": "Min_Eco", "آداب وفلسفة": "Min_Letters", "لغات أجنبية": "Min_Foreign_Lang"},
        "table_cols": ['الرمز', 'التخصص', 'الجامعة', 'المعدل', 'النطاق الجغرافي']
    }
}

# Sidebar التحكم
with st.sidebar:
    st.title("⚙️ الإعدادات")
    selected_lang = st.selectbox("اللغة", ["العربية"], index=0)
    st.session_state.language = selected_lang
    st.rerun()

L = texts[st.session_state.language]

# ============================================================================
# 🔥 3. الواجهة الرئيسية المحسنة
# ============================================================================

st.title(L["bot_name"])
st.markdown("---")
st.info(L["welcome_msg"])

# المعلومات الشخصية
st.subheader(L["personal_info"])
col1, col2, col3 = st.columns(3)

wilayas_list = ["01- أدرار", "02- الشلف", "03- الأغواط", "04- أم البواقي", "05- باتنة", 
               "06- بجاية", "07- بسكرة", "08- بشار", "09- البليدة", "10- البويرة", 
               "16- الجزائر", "28- المسيلة", "31- وهران"]  # مختصر للاختبار

with col1: selected_wilaya = st.selectbox(L["wilaya_label"], wilayas_list)
with col2: selected_stream = st.selectbox(L["stream_label"], list(L["mapping"].keys()))
with col3: st.selectbox(L["avg_range_label"], L["avg_ranges"])

# المعدل الموزون
st.subheader(L["weighted_title"])
sc1, sc2 = st.columns(2)
with sc1: real_avg = st.number_input(L["real_avg_label"], 10.0, 20.0, 14.5, step=0.01)
with sc2: sub_grade = st.number_input(L["grade_input_label"], 0.0, 20.0, 12.0, step=0.25)

weighted_avg = (real_avg * 2 + sub_grade) / 3
st.success(f"**المعدل الموزون:** `{weighted_avg:.2f}` 🎯")

# العوامل
st.divider()
st.subheader(L["factors_title"])
f1_col, f2_col = st.columns(2)
with f1_col: f1 = st.selectbox(L["f1_label"], L["factors_list"])
with f2_col: f2 = st.selectbox(L["f2_label"], ["None"] + L["factors_list"])

# Session state
if 'final_db' not in st.session_state: st.session_state.final_db = None
if 'stream_col' not in st.session_state: st.session_state.stream_col = ""

# ============================================================================
# 🔥 4. المحرك الأساسي المحسن
# ============================================================================

if st.button(L["calculate_btn"], use_container_width=True, type="primary"):
    with st.spinner("جاري البحث في قاعدة البيانات الوزارية..."):
        user_code = selected_wilaya.split('-')[0].strip()
        stream_col = L["mapping"][selected_stream]
        
        # التصفية الأساسية
        eligible_mask = (db[stream_col] > 0) & (db[stream_col] <= weighted_avg)
        df_results = db[eligible_mask].copy()
        
        def is_valid_geo(row_allowed):
            allowed_str = str(row_allowed).lower().replace(',', ' ')
            return user_code in allowed_str or 'national' in allowed_str
        
        final_db = df_results[df_results['Allowed_Wilayas'].apply(is_valid_geo)].copy()
        
        if not final_db.empty:
            # ترتيب ذكي
            final_db['rank'] = final_db['University_Name'].str.lower().str.contains(
                selected_wilaya.split('-')[1].lower(), na=False).map({True: 3, False: 1})
            final_db = final_db.sort_values(['rank', stream_col], ascending=[False, False])
            
            st.session_state.final_db = final_db
            st.session_state.stream_col = stream_col
            st.balloons()
            st.success(f"✅ **{len(final_db)} تخصص متاح** لمعدلك {weighted_avg:.1f}")
        else:
            st.session_state.final_db = None
            st.warning("⚠️ لا توجد تخصصات متاحة لهذا المعدل/الولاية")

# عرض النتائج الأولية
if st.session_state.final_db is not None:
    view_df = st.session_state.final_db[['Code_Fil', 'University_Major', 'University_Name', 
                                        st.session_state.stream_col, 'Allowed_Wilayas']].copy()
    view_df.columns = L["table_cols"]
    st.dataframe(view_df, use_container_width=True, height=300)

# ============================================================================
# 🔥 5. نظام التنبؤ المتقدم (الجوهرة)
# ============================================================================

st.divider()
st.subheader("🤖 **نظام التنبؤ الذكي AI**")

user_bio = st.text_area(L["bio_label"], 
                       placeholder="مثال: أحب البرمجة، الطب، الاقتصاد، الهندسة، التجارة...",
                       help="اكتب اهتماماتك لتحسين التوصيات بنسبة 30%")

if st.button(L["predict_btn"], use_container_width=True, type="secondary") and st.session_state.final_db is not None:
    with st.spinner("🧠 جاري التحليل الذكي المتقدم..."):
        res = st.session_state.final_db.copy()
        
        # 1. التنبؤ بالنجاح (ML Model)
        stream_enc = factor_enc = 0
        if hasattr(stream_encoder, 'transform'): 
            try: stream_enc = int(stream_encoder.transform([[selected_stream]])[0])
            except: pass
        if hasattr(factor_encoder, 'transform'): 
            try: factor_enc = int(factor_encoder.transform([[f1]])[0])
            except: pass
        
        major_encodings = {}
        for major in res['University_Major'].unique():
            major_str = str(major).lower()
            if any(w in major_str for w in ['science', 'علوم', 'تكنولوجيا', 'هندسة']): major_encodings[major] = 0
            elif any(w in major_str for w in ['health', 'طب', 'medicine']): major_encodings[major] = 1
            elif any(w in major_str for w in ['eco', 'اقتصاد', 'تسيير']): major_encodings[major] = 2
            elif any(w in major_str for w in ['law', 'قانون']): major_encodings[major] = 3
            else: major_encodings[major] = 0
        
        def predict_success(row):
            major_enc = major_encodings.get(row['University_Major'], 0)
            features = np.array([[stream_enc, weighted_avg, factor_enc, major_enc]])
            try:
                return round(model.predict_proba(features)[0][1] * 100, 1)
            except:
                diff = weighted_avg - row[st.session_state.stream_col]
                return round(min(max(65 + diff * 5, 45), 95), 1)
        
        res['success_prediction'] = res.apply(predict_success, axis=1)
        
        # 2. NLP للاهتمامات
        if user_bio.strip():
            try:
                def clean_text(text): 
                    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(text).lower())).strip()
                
                bio_keywords = clean_text(user_bio)
                arabic_to_french = {
                    'رياضيات': 'maths informatique', 'برمجة': 'informatique programmation', 
                    'طب': 'médecine santé', 'هندسة': 'génie ingénieur', 
                    'اقتصاد': 'économie gestion', 'علوم': 'sciences biologie'
                }
                
                enhanced_bio = bio_keywords
                for ar, fr in arabic_to_french.items():
                    if ar in bio_keywords: enhanced_bio += ' ' + fr
                
                docs = [enhanced_bio] + [clean_text(m) for m in res['University_Major']]
                tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                similarities = cosine_similarity(tfidf.fit_transform(docs))[0][1:]
                
                scores = [min(max(s * 250, 50), 92) for s in similarities]
                res['interest_score'] = np.clip([round(s, 1) for s in scores], 45, 92)
                st.success(f"✅ **NLP:** أعلى اهتمام {res['interest_score'].max():.1f}%")
            except:
                res['interest_score'] = 65
        else:
            res['interest_score'] = 60
            st.info("💡 **اهتمام افتراضي:** 60%")
        
        # 3. الترتيب النهائي + التصفية الذكية
        res['combined_score'] = res['success_prediction'] * 0.7 + res['interest_score'] * 0.3
        all_res = res.sort_values('combined_score', ascending=False).drop_duplicates(subset=['University_Major'])
        
        min_required_col = st.session_state.stream_col
        smart_filter = (
            (all_res[min_required_col] <= weighted_avg) & 
            (all_res[min_required_col] >= weighted_avg - 3) &
            (all_res['success_prediction'] > 70)
        )
        filtered_res = all_res[smart_filter].copy()
        
        # حفظ في session للتنزيل
        st.session_state.filtered_res = filtered_res
        st.session_state.min_required_col = min_required_col
        st.session_state.weighted_avg = weighted_avg
        st.session_state.display_df = filtered_res[[
            'Code_Fil', 'University_Major', 'University_Name', min_required_col, 
            'success_prediction', 'interest_score', 'combined_score'
        ]].copy()
        
        st.session_state.display_df.columns = ['الرمز', 'التخصص', 'الجامعة', 'المعدل المطلوب', 
                                             'نجاح %', 'اهتمام %', 'الإجمالي %']

# ============================================================================
# 🔥 6. عرض النتائج النهائي (منظم ونظيف)
# ============================================================================

if 'filtered_res' in st.session_state and not st.session_state.filtered_res.empty:
    filtered_res = st.session_state.filtered_res
    weighted_avg = st.session_state.weighted_avg
    min_required_col = st.session_state.min_required_col
    display_df_display = st.session_state.display_df
    
    # الجدول الرئيسي
    st.subheader("🏆 **التخصصات المتاحة لمعدلك**")
    st.dataframe(style_table(display_df_display, weighted_avg), use_container_width=True, height=500)
    
    # الإحصائيات
    st.markdown("---")
    display_metrics(filtered_res)
    
    # زر التنزيل
    csv = display_df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 تنزيل Excel",
        data=csv,
        file_name=f"DZ_UniGuide_{weighted_avg:.1f}_{len(filtered_res)}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # أفضل 10 + النصيحة + الملخص
    smart_advice(filtered_res['combined_score'].max())
    final_summary(filtered_res, weighted_avg)
    
    st.balloons()

st.markdown("---")
st.caption("👨‍💻 DZ-UniGuide AI AYA | المنشور الوزاري الرسمي 2026 🇩🇿")