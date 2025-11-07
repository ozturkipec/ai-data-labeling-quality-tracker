# AI Data Labeling Quality Tracker

An interactive **Streamlit** app that helps analyze and visualize AI labeling quality.  
Upload a CSV file with model predictions and true labels — and instantly view accuracy, precision, recall, F1-score, confusion matrix, and error breakdowns.

---

##  Features
- Upload CSV with `true_label` and `predicted_label`
- View **accuracy**, **precision**, **recall**, and **F1-score**
- Explore **false positives** and **false negatives** for each label
- Inspect confusion matrix and per-label metrics
- Fully interactive dashboard built in **Python + Streamlit**

---

## 📂 Project Structure
```
ai-data-labeling-quality-tracker/
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
├── example_data.csv     # Sample dataset
└── README.md            # Project documentation
```
---

## 📊 Example Data
The app expects at least two columns:
- `true_label` — ground truth label  
- `predicted_label` — model’s predicted label  

Example (from `example_data.csv`):
```csv
id,true_label,predicted_label,confidence,notes
1,person,person,0.98,correct detection
2,vehicle,person,0.62,misclassified
3,person,person,0.91,correct detection

How to Run Locally

1. Install dependencies (pip install -r requirements.txt) 
2. Run the app ( streamlit run app.py ) 
3. Open your browser


