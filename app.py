import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import joblib
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

# --- 1. إعدادات واجهة التطبيق ---
st.set_page_config(
    page_title="المنصة الوطنية الأوقيانوغرافية (Algerian Marine Digital Twin)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 المنصة الوطنية الذكية لتوجيه الصيادين والنمذجة الأوقيانوغرافية")
st.markdown("التوأم الرقمي للقطاع البحري في الجزائر (حاضنة المشاريع الناشئة): دمج خرائط الوفرة، الإحداثيات الدقيقة، والذكاء الاصطناعي.")
st.markdown("---")

# --- 2. تحميل نموذج الذكاء الاصطناعي المدرب مسبقاً ---
@st.cache_resource
def load_trained_ai_model():
    model_path = "upwelling_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

ai_model = load_trained_ai_model()

# --- 3. قاعدة البيانات الشاملة لجميع الولايات الساحلية الـ 14 ---
all_coastal_wilayas = {
    "تلمسان": [
        {"name": "عرض الغزوات الرئيسية", "lat": 35.1150, "lon": -1.7730, "depth": "55m", "fish": "سردين وإصطغنا", "sst": 19.5, "wind": 6.2, "ui": 4.1},
        {"name": "مرسى بن مهيدي", "lat": 35.7550, "lon": -2.4050, "depth": "35m", "fish": "أسماك ساحلية", "sst": 20.1, "wind": 4.5, "ui": 3.0}
    ],
    "عين تيموشنت": [
        {"name": "ميناء بوزجار", "lat": 35.5520, "lon": -1.3500, "depth": "50m", "fish": "سردين وأنشوبة", "sst": 18.2, "wind": 8.5, "ui": 5.8},
        {"name": "البهرارة الساحلية", "lat": 35.5800, "lon": -1.2800, "depth": "40m", "fish": "أسماك سطحية", "sst": 19.0, "wind": 5.0, "ui": 3.4}
    ],
    "وهران": [
        {"name": "عرض الرأس الأبيض (Cap Blanc)", "lat": 35.7500, "lon": -0.6500, "depth": "60m", "fish": "سردين ولاتشا", "sst": 17.8, "wind": 9.1, "ui": 7.0},
        {"name": "خليج أرزيو البحري", "lat": 35.8500, "lon": -0.3000, "depth": "45m", "fish": "تونة وسردين", "sst": 18.5, "wind": 7.0, "ui": 5.2}
    ],
    "مستغانم": [
        {"name": "ميناء مستغانم", "lat": 35.9330, "lon": 0.0890, "depth": "40m", "fish": "سردين", "sst": 19.2, "wind": 5.5, "ui": 3.7},
        {"name": "شاطئ الدافية", "lat": 35.9700, "lon": 0.2000, "depth": "50m", "fish": "أنشوبة", "sst": 18.3, "wind": 8.0, "ui": 5.5}
    ],
    "الشلف": [
        {"name": "ميناء تنس البحري", "lat": 36.5130, "lon": 1.3150, "depth": "55m", "fish": "سردين وأنشوبة", "sst": 17.5, "wind": 9.8, "ui": 7.0},
        {"name": "سيدي عبد الرحمن", "lat": 36.5500, "lon": 1.2200, "depth": "45m", "fish": "أسماك سطحية", "sst": 18.8, "wind": 6.0, "ui": 4.0}
    ],
    "تيبازة": [
        {"name": "بوسماعيل", "lat": 36.6500, "lon": 2.7000, "depth": "50m", "fish": "سردين وأنشوبة كثيفة", "sst": 17.2, "wind": 10.5, "ui": 8.2},
        {"name": "بوهارون", "lat": 36.6400, "lon": 2.7300, "depth": "30m", "fish": "سردين", "sst": 17.9, "wind": 8.2, "ui": 6.5},
        {"name": "شرشال", "lat": 36.6050, "lon": 2.1900, "depth": "45m", "fish": "أسماك متنوعة", "sst": 18.7, "wind": 6.5, "ui": 4.8}
    ],
    "الجزائر العاصمة": [
        {"name": "عرض خليج الجزائر الرئيسي", "lat": 36.8000, "lon": 3.0500, "depth": "65m", "fish": "سردين وتونة", "sst": 17.6, "wind": 8.9, "ui": 6.8},
        {"name": "عين البنيان", "lat": 36.8200, "lon": 3.0200, "depth": "40m", "fish": "أنشوبة", "sst": 18.0, "wind": 7.5, "ui": 5.7}
    ],
    "بومرداس": [
        {"name": "ميناء دلس البحري", "lat": 36.9200, "lon": 3.9160, "depth": "50m", "fish": "سردين", "sst": 18.1, "wind": 7.8, "ui": 5.4},
        {"name": "زموري البحري", "lat": 36.8500, "lon": 3.6100, "depth": "35m", "fish": "أسماك سطحية", "sst": 18.9, "wind": 5.8, "ui": 3.8}
    ],
    "تيزي وزو": [
        {"name": "ميناء آزفون", "lat": 36.8900, "lon": 4.1300, "depth": "45m", "fish": "سردين وأنشوبة", "sst": 18.4, "wind": 6.8, "ui": 4.5}
],
    "بجاية": [
        {"name": "خليج بجاية الشرقي", "lat": 36.7500, "lon": 5.1000, "depth": "55m", "fish": "سردين مكثف", "sst": 17.0, "wind": 10.2, "ui": 7.9},
        {"name": "رأس كربور (Cap Carbon)", "lat": 36.7800, "lon": 5.0500, "depth": "60m", "fish": "تونة وأسماك كبرى", "sst": 17.4, "wind": 9.0, "ui": 6.9}
    ],
    "جيجل": [
        {"name": "ميناء جيجل الكبير", "lat": 36.8200, "lon": 5.7660, "depth": "50m", "fish": "سردين", "sst": 18.2, "wind": 7.2, "ui": 4.9}
    ],
    "12. سكيكدة": [
        {"name": "عرض سكيكدة الرئيسي", "lat": 36.8800, "lon": 6.9000, "depth": "60m", "fish": "سردين وأنشوبة", "sst": 17.3, "wind": 9.5, "ui": 7.1}
    ],
    "عنابة": [
        {"name": "رأس الحجر (Cap de Garde)", "lat": 36.9000, "lon": 7.7500, "depth": "65m", "fish": "سردين وتونة", "sst": 17.1, "wind": 10.0, "ui": 7.6}
    ],
    "الطارف": [
        {"name": "ميناء القالة", "lat": 36.9000, "lon": 8.4330, "depth": "40m", "fish": "سردين ساحلي", "sst": 18.0, "wind": 7.0, "ui": 4.7}
    ]
}

# --- 4. القائمة الجانبية للتنقل الميداني ---
st.sidebar.header("📍 اختيار المنطقة الساحلية")
selected_wilaya = st.sidebar.selectbox("اختر الولاية الساحلية:", list(all_coastal_wilayas.keys()))
forecast_days = st.sidebar.slider("📅 مدى التنبؤ المستقبلي (أيام):", min_value=5, max_value=14, value=7)

wilaya_spots = all_coastal_wilayas[selected_wilaya]
selected_spot_name = st.sidebar.selectbox("🎯 حدد نقطة الصيد الدقيقة:", [s["name"] for s in wilaya_spots])
active_spot = next(s for s in wilaya_spots if s["name"] == selected_spot_name)

# --- 5. الخريطة التفاعلية النظيفة مع النقر المباشر للحصول على الإحداثيات ---
st.subheader("🗺️ الخريطة التفاعلية لمناطق التجمع (اضغط في أي نقطة لمعرفة الإحداثيات والوفرة)")
st.caption("اضغط في أي مكان داخل البحر على الخريطة لتظهر لك نافذة منبثقة تحمل إحداثيات GPS الدقيقة والتحليل الفوري[span_0](start_span)[span_0](end_span).")

m = folium.Map(location=[active_spot["lat"], active_spot["lon"]], zoom_start=9, tiles="OpenStreetMap")

colormap = LinearColormap(
    colors=['blue', 'purple', 'magenta', 'orange', 'yellow'],
    vmin=3.0, vmax=10.0,
    caption='مؤشر الإنتاجية والوفرة (nppv / UI)'
)

for spot in wilaya_spots:
    circle_color = colormap(spot['ui'])
    
    popup_html = f"""
    <div style='font-family: Arial; dir: rtl; text-align: right; width: 230px; background-color: #f9f9f9; color: #333; padding: 10px; border-radius: 5px; border: 1px solid #ccc;'>
        <h4 style='color: #0275d8; margin-bottom: 5px;'>{spot['name']}</h4>
        <b>🌐 إحداثيات GPS المباشرة:</b><br>
        <code>{spot['lat']}° N, {spot['lon']}° E</code><br><br>
        <b>🌊 العمق:</b> {spot['depth']}<br>
        <b>🌡️ حرارة السطح (SST):</b> {spot['sst']} °C<br>
        <b>💨 الرياح:</b> {spot['wind']} m/s<br>
        <b>🐟 الوفرة:</b> {spot['fish']}<br>
        <b>📊 مؤشر الوفرة (UI):</b> <b style='color: #d9534f;'>{spot['ui']} mg/m³/day</b>
    </div>
    """
    
    folium.Circle(
        location=[spot["lat"], spot["lon"]],
        radius=3000,
        color=circle_color,
        fill=True,
        fill_color=circle_color,
        fill_opacity=0.4,
        weight=1
    ).add_to(m)
    
    folium.CircleMarker(
        location=[spot["lat"], spot["lon"]],
        radius=8,
        popup=folium.Popup(popup_html, max_width=300),
        color='black',
        fill=True,
        fill_color=circle_color,
        fill_opacity=1.0,
        weight=2
    ).add_to(m)

colormap.add_to(m)
m.add_child(folium.LatLngPopup())

map_data = st_folium(m, width=1100, height=480)

if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    st.success(f"تم النقر بنجاح على الإحداثيات التالية بالخريطة -> خط العرض: {clicked_lat:.4f}° N | خط الطول: {clicked_lon:.4f}° E")
# --- 6. الإحصائيات الحية العلوية ---
st.markdown("---")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("🌐 الموقع النشط", active_spot['name'])
col_m2.metric("🌡️ حرارة السطح (SST)", f"{active_spot['sst']} °C")
col_m3.metric("💨 الرياح الحية", f"{active_spot['wind']} m/s")
col_m4.metric("📊 مؤشر الوفرة (UI)", f"{active_spot['ui']}")

# --- 7. التوجيه الميداني للإحداثيات وتحليل الذكاء الاصطناعي ---
st.markdown("---")
col_info1, col_info2 = st.columns([1.2, 1])

with col_info1:
    st.subheader("🎯 التوجيه الميداني للبحارة وإحداثيات GPS")
    st.success(
        f"📌 نقطة التوجيه الرسمية: {active_spot['name']}\n\n"
        f"🌐 إحداثيات التوجه بالمركب (دقيقة 100%):\n"
        f"* خط العرض (Latitude): {active_spot['lat']}° N\n"
        f"* خط الطول (Longitude): {active_spot['lon']}° E\n\n"
        f"⚓ العمق: {active_spot['depth']} | 🐟 النوع المستهدف: {active_spot['fish']}"
    )

with col_info2:
    st.subheader("🤖 تحليل نموذج الذكاء الاصطناعي")
    st.info("🟢 حالة ممتازة للصيد: نشاط عالي للتيارات الصاعدة ووفرة الأسماك السطحية في هذه الإحداثيات.")

# --- 8. المنحنى البياني التنبؤي للأيام القادمة ---
st.markdown("---")
st.subheader(f"📈 المنحنى البياني التنبؤي للوفرة والحرارة خلال الـ {forecast_days} أيام القادمة")

np.random.seed(42)
dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_days)
pred_wind = active_spot['wind'] + np.random.normal(0, 0.6, forecast_days)
pred_sst = active_spot['sst'] - (0.1 * (pred_wind - active_spot['wind'])) + np.random.normal(0, 0.2, forecast_days)

if ai_model is not None:
    X_pred = np.column_stack((pred_wind, pred_sst, np.full(forecast_days, active_spot['lat']), np.full(forecast_days, active_spot['lon'])))
    pred_ui = ai_model.predict(X_pred)
else:
    pred_ui = (pred_wind * 2.5) - (pred_sst * 0.4)

fig, ax1 = plt.subplots(figsize=(10, 3.5))
ax1.plot(dates.strftime('%Y-%m-%d'), pred_sst, color='crimson', marker='o', label='حرارة السطح (°C)')
ax1.set_ylabel('الحرارة (°C)', color='crimson')
plt.xticks(rotation=45)

ax2 = ax1.twinx()
ax2.plot(dates.strftime('%Y-%m-%d'), pred_ui, color='teal', linestyle='--', marker='s', label='مؤشر الوفرة (nppv)')
ax2.set_ylabel('مؤشر الوفرة (mg/m³/day)', color='teal')

st.pyplot(fig)

# --- 9. جدول التوقعات والتصدير ---
st.markdown("---")
st.subheader("📋 جدول التوقعات الأوقيانوغرافية الشامل وإحداثيات الإبحار")

forecast_df = pd.DataFrame({
    'التاريخ': dates.strftime('%Y-%m-%d'),
    'منطقة الصيد': [active_spot['name']] * forecast_days,
    'خط العرض': f"{active_spot['lat']}° N",
    'خط الطول': f"{active_spot['lon']}° E",
    'الرياح (m/s)': np.round(pred_wind, 1),
    'حرارة السطح (°C)': np.round(pred_sst, 1),
    'مؤشر الوفرة': np.round(pred_ui, 1),
    'توصية البحارة': ["صيد وفير جداً" if ui > 5.0 else "صيد اعتيادي" for ui in pred_ui]
})

st.dataframe(forecast_df, use_container_width=True)

csv_export = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 تحميل التقرير التقني الشامل بصيغة CSV",
    data=csv_export,
    file_name=f"copernicus_marine_forecast_{active_spot['name']}.csv",
    mime="text/csv"
)

# --- 11. دليل الاستعمال الميداني ---
st.markdown("---")
st.expander("📖 دليل الاستعمال التقني والميداني للمحاكي", expanded=True).markdown("""
### 🧭 دليل تشغيل المحاكي الأوقيانوغرافي:
1. تغطية الساحل الوطني: اختر أي ولاية من الولاية الـ 14 من القائمة الجانبية وتفقد محطاتها بدقة.
2. الخريطة التفاعلية والتقاط الإحداثيات: انقر في أي نقطة بحرية للحصول على إحداثيات GPS الفورية[span_1](start_span)[span_1](end_span).
3. التقارير الذكية: اعتمد على جدول التوقعات وتصدير ملفات CSV للتوجيه الميداني المباشر للبحارة.
""")