# InsightDesk AI — Ready-to-Publish Package

This folder contains **adjusted, production-ready files** for your GitHub repository `FUTURE_ML_02`. Use these files to replace or supplement your existing project files.

## 📁 What's Included

1. **index.html** — Static interactive dashboard (no dependencies, works with Python's `http.server`)
2. **app/app.py** — Optional Streamlit dashboard (if you want a Python-based UI)
3. **requirements.txt** — Python dependencies for the Streamlit app (optional)
4. **run_app.bat** — Windows quick-start batch script

## 🚀 Quick Start (Static Dashboard — Recommended)

The **fastest way** to get a working dashboard is to use the static HTML version:

```powershell
# From project root, start a simple HTTP server
py -3 -m http.server 3000 --bind 127.0.0.1

# Open in browser
http://localhost:3000/prepared_repo/
```

**Features:**
- 📊 Category & Priority distribution charts
- 🔍 Search & filter tickets
- 📤 Upload CSV files
- ✅ No dependencies, no compilation, no build step needed

## 🎯 Alternative: Streamlit Dashboard

If you prefer a full Python Streamlit app:

```powershell
# Install dependencies (one-time)
pip install streamlit pandas plotly scikit-learn joblib

# Run the Streamlit app
streamlit run prepared_repo/app/app.py
```

**Features:**
- 📊 Interactive exploratory data analysis
- 🤖 Train ML models on your dataset
- 🎯 Single & batch predictions
- 💾 Download predictions as CSV

## 📋 Setup Instructions for GitHub

### Option A: Use the Static Dashboard

1. Copy **index.html** to your repo root or a `dashboard/` folder
2. Ensure your CSV files are in the repo root or a `data/` folder
3. Anyone can run:
   ```bash
   python -m http.server 3000 --bind 127.0.0.1
   # then open http://localhost:3000/dashboard/index.html (or wherever you placed it)
   ```

### Option B: Use Streamlit

1. Copy **app/**, **requirements.txt**, and **run_app.bat** into your repo root
2. Update `requirements.txt` with any additional dependencies
3. Users clone your repo and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app/app.py
   ```
   Or on Windows, double-click **run_app.bat**

## 📝 Customization

### For the Static Dashboard (index.html)
- Edit the colors by changing `#667eea` (purple) and `#764ba2` (dark purple) to your brand colors
- Modify CSV file paths in the `fetchData()` function if your data is in a different location
- Add more charts or tabs as needed (uses Plotly.js)

### For the Streamlit App (app.py)
- Update column names in `load_dataset()` to match your CSV headers
- Add more visualizations or ML models in the `main()` function
- Customize the sidebar and styling using Streamlit's theming

## 🔗 Dataset Paths

The apps look for CSV files in this order:
1. `customer_support_tickets.csv` (repo root)
2. `all_tickets_processed_improved_v3.csv` (repo root)
3. `data/customer_support_tickets.csv` (if you have a `data/` folder)

If your CSV is in a different location, edit the file paths in the code.

## 📦 Deployment

Both dashboards can be deployed easily:
- **Static (HTML):** Host on GitHub Pages, Vercel, Netlify, or any web server
- **Streamlit:** Deploy free on [Streamlit Cloud](https://share.streamlit.io/) — just connect your GitHub repo

## 🤝 Questions?

Feel free to open an issue or discussion in your repository. Both the static and Streamlit versions are designed to be simple, maintainable, and extendable.

---

**Created March 4, 2026** | Ready for GitHub 🎉
