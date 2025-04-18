Cricklytics - AI-Powered Cricket Analytics

Cricklytics is a data-driven cricket analytics platform focused on intelligent player evaluation, role classification, performance prediction, and injury forecasting. Designed with domestic and league-level scouting in mind, Cricklytics brings together machine learning and real-time data to support decision-making for analysts and team managers.

Features

- Player role classification using Random Forest and Neural Network models
- K-Means clustering for unsupervised role assignment
- Probability-based soft role boundaries instead of rigid thresholds
- AI-based injury prediction using player workload and recent performance trends
- Scouting dashboard with analytics for domestic-level players
- Real-time score integration using CricAPI (work in progress)

AI & ML Stack

- Supervised models: Random Forest, Neural Networks
- Unsupervised learning: K-Means clustering, PCA for dimensionality reduction
- Feature engineering includes runs, strike rate, average, matches played
- Weighted features and filters for low-experience players

Tech Stack

- Frontend: Streamlit
- Backend: Python with Pandas, Scikit-learn, TensorFlow, Keras
- APIs: CricAPI for live match data
- Dataset: 4000+ T20 players across leagues

Installation

Clone the repository and install dependencies

  git clone https://github.com/Pratyush038/cricklytics.git
  cd cricklytics
  pip install -r requirements.txt
  streamlit run app.py

