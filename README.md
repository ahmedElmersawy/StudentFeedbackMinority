# Feedback Atlas — training + demo

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Train on your CSV

Default run on `studentdataset.csv`:

```bash
python studentfeedback_analysis.py
```

If your dataset has a different schema, pass columns explicitly:

```bash
python studentfeedback_analysis.py --csv-path your_data.csv --text-cols col1 col2 --label-col target
```

Model artifacts are saved in `final_feedback_classifier/`.

## 3) Launch demo website

```bash
streamlit run app.py
```

In the app:
- Load model folder (`final_feedback_classifier`)
- Upload any CSV
- Choose one or more text columns
- Run predictions and download results
