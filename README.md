\# YES Bank Stock Price Prediction



A full-stack machine learning application for predicting the monthly closing price of YES Bank using historical stock-market data, engineered technical indicators, and a Linear Regression model.



The project combines a Python machine-learning pipeline with a FastAPI backend and a React/Vite interactive dashboard.



\---



\## 📌 Project Overview



This project explores whether historical market information can be used to predict the closing price of YES Bank.



The machine-learning pipeline includes:



\- Data preprocessing

\- Feature engineering

\- Technical indicators

\- Correlation analysis

\- Multicollinearity analysis using VIF

\- Linear Regression

\- Random Forest

\- Model comparison

\- Residual analysis

\- Final model evaluation

\- Interactive prediction through a web dashboard



The final selected model is \*\*Linear Regression V3\*\*, which uses seven input features.



\---





\##🧠 Machine Learning Model



The final model is:



Linear Regression V3



Input Features

Open

High

Low

Previous Close

Return

MA\_3

Volatility\_3

Target

Close



The dataset is divided chronologically into training and testing data to preserve the time-series order.



Dataset Split

Dataset	Samples

Training	145

Testing	37

Total	182



\##📊 Model Performance



The project compares multiple machine-learning approaches.



Model	Features	MAE	MSE	RMSE	R²

Linear Regression	Open, High, Low	9.7843	250.3307	15.8218	0.9848

Random Forest	Open, High, Low	13.9084	461.2433	21.4766	0.9719

Linear Regression V2	Open, High, Low, Previous\_Close	10.4629	273.5210	16.5385	0.9833

Random Forest V2	Open, High, Low, Previous\_Close	14.1890	556.5005	23.5903	0.9661

Linear Regression V3	Technical Features	9.8376	245.8189	15.6786	0.9850

Random Forest V3	Technical Features	14.3017	529.1807	23.0039	0.9678

Linear Regression V3

MAE  : 9.8376

MSE  : 245.8189

RMSE : 15.6786

R²   : 0.9850



\##📈 Technical Features

Previous Close



The previous month's closing price is used as an additional market-history feature.



Return



Monthly percentage return:



Return = (Current Close - Previous Close) / Previous Close

Moving Average



A 3-period moving average is calculated using:



MA\_3 = Rolling Mean of Close over 3 periods

Volatility



3-period rolling standard deviation of returns:



Volatility\_3 = Standard Deviation of Return over 3 periods



\##🔬 Feature Analysis

Correlation Analysis



The analysis showed strong relationships between the closing price and several price-based features.



Correlation with Close:



Feature	Correlation

Low	0.9953

MA\_3	0.9872

High	0.9848

Previous\_Close	0.9780

Open	0.9776

Return	0.0430

Volatility\_3	-0.1728

VIF Analysis



Variance Inflation Factor was used to identify multicollinearity among price-related features.



Feature	VIF

Open	81.93

High	79.20

Low	34.56



The high VIF values indicate strong multicollinearity between several price-based variables.



\##📉 Residual Analysis



For Linear Regression V3:



Mean Residual       : -5.0423

Minimum Residual    : -62.9409

Maximum Residual    : 23.3998

Residual Std Dev    : 14.8457



The model's residual analysis helps identify the difference between actual and predicted closing prices and provides additional information about prediction error.



\##🖥️ Interactive Dashboard



The project includes a professional React dashboard.



The dashboard provides:



Model status

Model performance metrics

Interactive prediction form

Seven technical feature inputs

Prediction result

Prediction timestamp

Model information

Historical actual-vs-predicted chart

Responsive dark financial dashboard design

Loading and error states

Example input values

Reset functionality

Example Prediction



Example input:



Open              = 320

High              = 335.50

Low               = 305

Previous Close    = 318.25

Return            = 0.0129

MA\_3              = 317.80

Volatility\_3      = 0.0450



The prediction is generated through the actual trained model and FastAPI backend.



\##⚡ FastAPI Backend



The backend is implemented using FastAPI.



Available Endpoints

Health Check

GET /health



Checks whether the backend is running.



Prediction

POST /predict



Accepts the seven model features and returns the predicted closing price.



Model Information

GET /model-info



Returns information about the trained model and its performance.



Historical Predictions

GET /predictions



Returns the 37 test-set actual and predicted values used by the dashboard chart.



\##📁 Project Structure

yes-bank-stock-prediction/

│

├── backend/

│   ├── \_\_init\_\_.py

│   └── main.py

│

├── data/

│   ├── yes\_bank\_stock.csv

│   ├── model\_comparison.csv

│   └── final\_prediction\_plot.png

│

├── frontend/

│   ├── public/

│   ├── src/

│   │   ├── App.jsx

│   │   ├── App.css

│   │   ├── index.css

│   │   └── main.jsx

│   ├── package.json

│   ├── package-lock.json

│   └── vite.config.js

│

├── models/

│   ├── linear\_regression\_v3.pkl

│   └── scaler\_v3.pkl

│

├── src/

│   ├── correlation\_analysis.py

│   ├── evaluate\_model.py

│   ├── feature\_engineering.py

│   ├── final\_comparison.py

│   ├── final\_model\_comparison.py

│   ├── final\_prediction\_plot.py

│   ├── final\_report.py

│   ├── linear\_regression\_v2.py

│   ├── linear\_regression\_v3.py

│   ├── model\_comparison.py

│   ├── predict.py

│   ├── prepare\_features.py

│   ├── prepare\_technical\_features.py

│   ├── random\_forest\_model.py

│   ├── random\_forest\_v2.py

│   ├── random\_forest\_v3.py

│   ├── residual\_analysis.py

│   ├── scale\_features.py

│   ├── scale\_technical\_features.py

│   ├── technical\_features.py

│   ├── train\_model.py

│   ├── vif\_analysis.py

│   └── visualization.py

│

├── .gitignore

├── requirements.txt

└── README.md



\##🛠️ Technologies Used

Machine Learning

Python

Pandas

NumPy

Scikit-learn

Joblib

Matplotlib

Backend

FastAPI

Uvicorn

Python

Frontend

React

Vite

JavaScript

CSS

Development

VS Code

Git

GitHub



\##⚙️ Installation

1\. Clone the repository

git clone https://github.com/Umaidenthusiast/yes-bank-stock-prediction.git

cd yes-bank-stock-prediction

2\. Create a virtual environment



Windows:



python -m venv venv

3\. Install Python dependencies

.\\venv\\Scripts\\python.exe -m pip install -r requirements.txt

4\. Start the FastAPI backend

.\\venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8001



Backend:



http://127.0.0.1:8001

5\. Start the frontend



Open another terminal:



cd frontend

npm install

npm run dev



Frontend:



http://localhost:5173



\## 🔐 Environment Configuration



The frontend uses a Vite environment variable for the backend URL.



Create:



frontend/.env



with:



VITE\_API\_URL=http://127.0.0.1:8001



The .env file is excluded from GitHub through .gitignore.



An example configuration is provided as:



frontend/.env.example



\##🧪 Running the Original ML Pipeline



Individual analysis and training scripts are available inside:



src/



Examples:



.\\venv\\Scripts\\python.exe src/prepare\_technical\_features.py

.\\venv\\Scripts\\python.exe src/linear\_regression\_v3.py

.\\venv\\Scripts\\python.exe src/residual\_analysis.py

.\\venv\\Scripts\\python.exe src/vif\_analysis.py

.\\venv\\Scripts\\python.exe src/final\_report.py



\##⚠️ Limitations



This project is intended as a machine-learning and software-engineering project rather than a financial trading system.



Important limitations include:



Historical data does not guarantee future performance.

Stock prices are influenced by many external factors not included in the model.

The model uses a relatively small monthly dataset.

Linear Regression assumes a linear relationship between the input features and target.

Strong multicollinearity exists among several price-related features.

The prediction error can become larger during periods of unusual market movement.



The predictions should therefore be interpreted as model outputs for educational and analytical purposes rather than financial advice.



\##🎯 Future Improvements



Potential future improvements include:



LSTM/GRU time-series models

XGBoost/LightGBM comparison

Hyperparameter optimization

More historical and higher-frequency data

Additional technical indicators

News and sentiment features

Automated model retraining

Model monitoring

Cloud deployment

User authentication

Prediction history

Interactive historical price charts



\##👨‍💻 Author



Mohammad Umaid Moulali Gudmithe



Computer Engineering Student

Pune, Maharashtra, India



\##📄 License



This project is intended for educational, research, and portfolio purposes.

