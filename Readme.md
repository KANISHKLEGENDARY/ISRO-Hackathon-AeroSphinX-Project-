# 🌍 AeroSphinx – PM2.5 Estimation & Forecasting

## 📌 Overview
AeroSphinx is a machine learning project developed for ISRO’s Bharatiya Antariksha Hackathon (BAH 2025).  
It focuses on estimating and forecasting surface-level PM2.5 concentrations in Faridabad using multi-source environmental data.

---

## 📊 Data Sources
- 🛰️ Satellite Data: INSAT AOD  
- 🌫️ Ground Monitoring: CPCB PM2.5  
- 🌦️ Reanalysis Data: NASA MERRA-2  

---

## 🚀 Key Features
- Integration of heterogeneous datasets (satellite + ground + weather)
- End-to-end ML pipeline for PM2.5 prediction
- Feature engineering to capture environmental interactions
- Short-term forecasting (1-day ahead prediction)
- Dashboard-ready output for visualization

---

## 🧠 Feature Engineering
We engineered additional features to improve model performance:

### 🔹 Temperature Difference
Captures atmospheric stability and vertical mixing, affecting pollutant dispersion.

### 🔹 Humidity Ratio
Represents interaction between moisture and temperature, influencing particle growth.

---

## ⏱️ Forecasting Strategy
- Predicts PM2.5 for the next day  
- Focused on short-term forecasting  
- Designed for evaluation and real-world applicability  

---

## 📈 Seasonal Insights
- ❄️ Winter: High PM2.5 due to low dispersion  
- 🌧️ Monsoon: Lower pollution due to washout effect  
- ☀️ Summer: Moderate levels influenced by wind and temperature  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Xarray  
- Matplotlib  
- Power BI (for visualization)  

---

## ⚙️ How to Run

```bash
python ML_Model.py