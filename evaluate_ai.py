import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# تحميل الموديل الذي حفظناه في ملف التدريب
model = joblib.load('university_model_smart.pkl')

# (هنا تضعين كود لرسم Confusion Matrix أو ROC Curve)
# هذا ما يملأ صفحات المذكرة بالنتائج العلمية!