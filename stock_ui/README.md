# Stock Dashboard

View `featured_data.csv` with candlestick charts, Core 3 panels (zscore_5, rsi_2, tr_rel_10), signal markers, “What happened next?”, and zscore_5 bucket distribution.

**Run from project root (parent of `stock_ui`):**

```bash
pip install -r stock_ui/requirements.txt
streamlit run stock_ui/app.py
```

Or from inside `stock_ui`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).
