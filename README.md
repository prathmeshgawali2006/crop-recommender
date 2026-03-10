\# 🌾 Crop Recommendation System



A machine learning project that recommends the best crops to grow based on soil nutrients and climate conditions — built to help small farmers make smarter, data-driven decisions.



---



\## 📌 Problem Statement



Small and rural farmers often lack access to agricultural expertise. Planting the wrong crop for their soil and climate leads to poor yields and financial loss. This tool uses machine learning to recommend the most suitable crop based on simple measurable inputs.



---



\## 💡 Solution



A Random Forest classifier trained on 2,200 soil and climate samples across 22 crop types. The model achieves \*\*99.32% accuracy\*\* and provides:

\- ✅ The best crop recommendation with confidence score

\- 📋 Top 3 alternative crops

\- 💡 A human-readable explanation for each recommendation



---



\## 📂 Project Structure

```

crop-recommender/

│

├── data/

│   ├── crop\_data.csv                # Dataset (2200 rows, 22 crops)

│   ├── crop\_distribution.png        # Chart: crop counts

│   ├── feature\_distributions.png    # Chart: feature distributions

│   ├── correlation\_heatmap.png      # Chart: feature correlations

│   └── feature\_importance.png       # Chart: what matters most

│

├── notebooks/

│   └── exploration.ipynb            # EDA notebook

│

├── src/

│   ├── train.py                     # Model training script

│   └── predict.py                   # CLI prediction tool

│

├── model/

│   └── crop\_model.pkl               # Saved trained model

│

├── requirements.txt                 # Python dependencies

├── .gitignore                       # Files excluded from Git

└── README.md                        # You are here

```



---



\## 🧪 Dataset



| Property     | Details                          |

|--------------|----------------------------------|

| Source       | Kaggle Crop Recommendation Dataset |

| Rows         | 2,200                            |

| Features     | 7 (N, P, K, temp, humidity, pH, rainfall) |

| Target       | Crop name (22 classes)           |

| Balance      | Perfectly balanced (100 per crop) |

| Missing Data | None                             |



\### Input Features



| Feature         | Description                        | Range        |

|-----------------|------------------------------------|--------------|

| N               | Nitrogen content in soil           | 0 – 140      |

| P               | Phosphorus content in soil         | 5 – 145      |

| K               | Potassium content in soil          | 5 – 205      |

| Temperature     | Average temperature in °C          | 8 – 44       |

| Humidity        | Relative humidity in %             | 14 – 100     |

| pH              | Soil pH level                      | 3.5 – 9.5    |

| Rainfall        | Annual rainfall in mm              | 20 – 300     |



---



\## 🤖 Model



| Property         | Details                          |

|------------------|----------------------------------|

| Algorithm        | Random Forest Classifier         |

| Trees            | 100                              |

| Train/Test Split | 80% / 20%                        |

| Training Samples | 1,760                            |

| Test Samples     | 440                              |

| Accuracy         | \*\*99.32%\*\*                       |



---



\## 🚀 How to Run



\### 1. Clone the Repository

```bash

git clone https://github.com/prathmeshgawali2006/crop-recommender.git

cd crop-recommender

```



\### 2. Create and Activate Virtual Environment

```bash

\# Windows

python -m venv venv

venv\\Scripts\\activate



\# Mac/Linux

python3 -m venv venv

source venv/bin/activate

```



\### 3. Install Dependencies

```bash

pip install -r requirements.txt

```



\### 4. Train the Model

```bash

python src/train.py

```



\### 5. Run the Prediction Tool

```bash

python src/predict.py

```




---
### 6. Open the Web Interface (Optional)

Simply open `index.html` in any browser — no server needed!
```bash
# Windows
start index.html

# Mac
open index.html
```

The web UI lets you enter values and get recommendations visually with confidence bars and alternative crops.

---


\## 🌱 Example Usage

```

====================================================

&nbsp;       🌾 CROP RECOMMENDATION SYSTEM

====================================================

&nbsp; Enter your soil and climate details below.



&nbsp; Nitrogen (N) (0–140): 90

&nbsp; Phosphorus (P) (5–145): 42

&nbsp; Potassium (K) (5–205): 43

&nbsp; Temperature (°C) (8–44): 21

&nbsp; Humidity (%) (14–100): 82

&nbsp; pH (3.5–9.5): 6.5

&nbsp; Rainfall (mm) (20–300): 202



====================================================

          🌱 RECOMMENDATION RESULTS

====================================================



&nbsp; ✅ BEST CROP  :  RICE

&nbsp; 📊 Confidence :  99.0%

&nbsp; 💡 Why        :  Thrives in high humidity and heavy rainfall.



&nbsp; 📋 ALTERNATIVES:



&nbsp; 2. Coconut (0.5%)

&nbsp;    Requires high humidity and heavy rainfall.



&nbsp; 3. Jute (0.3%)

&nbsp;    Grows best in warm, humid, rainy conditions.



====================================================

```



---



\## 📊 Key Insights from EDA



\- \*\*Rainfall\*\* and \*\*humidity\*\* are the strongest predictors of crop type

\- \*\*pH\*\* has the least impact on predictions

\- The dataset is perfectly balanced — no bias toward any crop

\- No missing values — clean, ready-to-use data



---



\## 🛠️ Tech Stack



| Tool            | Purpose                     |

|-----------------|-----------------------------|

| Python 3        | Core language               |

| pandas          | Data loading and analysis   |

| scikit-learn    | Machine learning model      |

| matplotlib      | Charts and visualizations   |

| seaborn         | Heatmap visualization       |

| joblib          | Saving and loading model    |

| Jupyter         | EDA notebook                |



---



\## 👤 Author



\*\*Prathmesh Gawali\*\*

\- GitHub: \[@prathmeshgawali2006](https://github.com/prathmeshgawali2006)



---



\## 📄 License



This project is open source and available under the \[MIT License](LICENSE).

```





