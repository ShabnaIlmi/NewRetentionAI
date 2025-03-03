# RetentionAI

**RetentionAI** is an AI-powered web application built using **Streamlit** that revolutionizes customer churn prediction for two key industries: **Telecom** and **Banking**. With cutting-edge **Machine Learning models**, RetentionAI empowers businesses to anticipate customer behavior, reduce churn rates, and make data-driven decisions. The system provides highly accurate predictions based on various customer attributes, helping organizations retain their valuable clientele.

## Features
- Telecom Customer Churn Prediction
- Banking Customer Churn Prediction
- Interactive Web Interface with Streamlit
- Real-time Predictions

## Tech Stack
- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Random Forest**

## Dataset
The datasets used to train the models can be found on Kaggle:
- [Telecom Churn Dataset](https://www.kaggle.com/code/ybifoundation/telecom-customer-churn-prediction)
- [Banking Customer Churn Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)

## Installation
1. Clone the repository:
   ```bash
   git clone (https://github.com/ShabnaIlmi/NewRetentionAI)
   cd RetentionAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Select the prediction model (Telecom or Banking).
2. Upload the customer dataset in **CSV** format.
3. Click **Predict** to generate churn predictions.

## Project Structure
```
RetentionAI/
│
├─ .devcontainer/      
├─ .git/               
├─ assets/             
├─ app.py              
├─ bank_churn_model.pkl 
├─ telecom_churn_model.pkl 
├─ scaler_bank.pkl      
├─ scaler_telecom.pkl  
├─ requirements.txt     
└─ README.md           
```

## License
This project is licensed under the **MIT License**.

## Contributors
- Fathima Shabna Ilmi

## Contact
If you have any questions, feel free to contact me at **Shabna Ilmi - ilmishabna03@gmail.com)**.

