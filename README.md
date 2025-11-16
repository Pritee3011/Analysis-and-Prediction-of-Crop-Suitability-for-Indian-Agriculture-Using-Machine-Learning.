# Analysis and Prediction of Crop Suitability for Indian Agriculture Using Machine Learning

A data-driven web application that uses a **Genetic Algorithm-optimized Random Forest model** to analyze agricultural data and recommend the most suitable crops for farmers in India based on soil and climate conditions.

**Live Project Website:** [**<-- Add Your Deployed Website Link Here -->**]

---

## 🎯 Project Objective

This study aims to move Indian agriculture toward a more sustainable, economically viable, and technology-driven approach. The primary objectives are:

* To develop a highly accurate **Genetic Algorithm-optimized Random Forest (GA-RF)** crop recommendation system using Indian agricultural data.
* To conduct an in-depth **statistical analysis** of crop price variations using One-Way and Two-Way ANOVA to understand economic factors.
* To identify the most **critical soil and environmental features** (e.g., Rainfall, N, P, K) that affect crop suitability.
* To provide a practical, interactive system that supports data-driven crop selection for both **farmers and policymakers**.

---

## 🚀 How to Run This Project

Follow these instructions to get the web application running on your local machine.

### Prerequisites

* Python 3.x
* pip (Python package installer)

### Installation & Execution

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    The core libraries needed are Flask (for the web server), scikit-learn (for the model), pandas, and numpy (for data handling).
    ```sh
    pip install Flask scikit-learn pandas numpy
    ```
    *(Note: You can also create a `requirements.txt` file with these package names and run `pip install -r requirements.txt`)*

4.  **Run the Flask application:**
    Once the libraries are installed, run the main Python script (e.g., `app.py`).
    ```sh
    python app.py
    ```

5.  **Access the application:**
    Open your web browser and navigate to the local address provided by Flask, typically:
    **`http://127.0.0.1:5000/`**

---

## 📊 Dataset Details

The model was trained on the **`indian_crop_new.csv`** dataset. This dataset contains a comprehensive collection of agricultural data, including:

* **Soil Nutrients:** `N` (Nitrogen), `P` (Phosphorus), `K` (Potassium).
* **Environmental Factors:** `temperature`, `humidity`, `rainfall`.
* **Soil Properties:** `ph` value.
* **Economic Data:** `Crop Prices`.
* **Target Variable:** `CROP` (the specific crop type, e.g., Rice, Maize, etc.).

Before training, the data underwent rigorous preprocessing, including outlier handling (clipping), skewness correction (log transformation), and standardization (z-score).

---

## 🤖 Algorithm/Model Used

The core of this project is a **Genetic Algorithm-optimized Random Forest (GA-RF) Classifier**.

* **Why Random Forest?**
    This algorithm was selected because it effectively handles complex, non-linear relationships between soil, climate, and crops. It is also highly resistant to overfitting and provides strong accuracy for multi-class classification problems like this one.

* **Why a Genetic Algorithm (GA)?**
    To maximize the model's performance, a GA was used for hyperparameter optimization. Instead of manual tuning, the GA automatically "evolves" the best combination of parameters (like `n_estimators`, `max_depth`). This metaheuristic approach searches the parameter space to find a set that results in the highest possible accuracy.

---

## 📈 Results

The final model and statistical analysis yielded significant insights.

### Model Performance

The GA-optimized Random Forest model achieved outstanding performance on the held-out test data:

* **Accuracy:** **99.55%**
* **Weighted F1-Score:** **0.9955**

The confusion matrix confirmed high per-class prediction accuracy with minimal misclassifications.



### Statistical Analysis

* **One-Way ANOVA:** A test on four legume crops (e.g., KidneyBeans, PigeonPeas) showed a highly significant difference in their mean crop prices (p < 0.0001). This confirms that crop choice is a critical economic decision.

* **Two-Way ANOVA:** An analysis of fruit prices across different states revealed a **significant interaction effect** (p = 0.0151) between `CROP` and `STATE`. This is a key finding: it proves that crop profitability is not uniform and depends on the specific combination of the crop and the region where it's grown.

### Feature Importance

The trained model identified the most influential factors for predicting crop suitability:

1.  **RAINFALL** (Rank 1)
2.  **N_SOIL** (Rank 2)
3.  **TEMPERATURE** (Rank 3)
4.  P_SOIL (Rank 4)
5.  K_SOIL (Rank 5)
6.  HUMIDITY (Rank 6)
7.  PH_SOIL (Rank 7)

This ranking aligns with established agronomic knowledge, confirming that rainfall and soil nitrogen are primary determinants of crop success.



---

## 🏁 Conclusion

This study successfully developed an integrated system for data-driven crop selection. By optimizing a Random Forest with a Genetic Algorithm, we created a highly accurate (99.55%) predictive model tailored to Indian agricultural data.

Furthermore, statistical analysis (ANOVA) confirmed that both crop choice and geographical location have a statistically significant impact on crop prices. This integration of a predictive ML model with statistical validation provides a robust tool for farmers and policymakers, addressing the critical need to balance both agronomic suitability and economic viability.

---

## 🔭 Future Scope

The current system is based on static, historical data. The future potential of this work lies in its evolution into a dynamic, real-time decision support system:

* **IoT Sensor Integration:** Connect with on-field IoT sensors to stream real-time soil data (N, P, K, pH, moisture) instead of manual entry.
* **Real-Time Forecasts:** Integrate with meteorological APIs to use short-term and long-term weather forecasts (rainfall, temperature) in the recommendation.
* **Advanced Economic Modeling:** Incorporate real-time market price data, transportation costs, and supply chain logistics to shift the recommendation from "what can I grow" to "what is the *most profitable* crop to grow right now".
* **Mobile Deployment & XAI:** Deploy the system as a user-friendly mobile application and use Explainable AI (XAI) to build trust by clearly explaining *why* a certain crop is recommended.

---

## 📚 References

The full list of academic references and citations is available in the project's technical paper.

---

### Project Team

* **Pritee Pardeshi**
* **Anjali Mhetre**
* **Bhumi Bhat**

*Department of Computer Engineering, MKSSS’s Cummins College Of Engineering for Women, Pune, India.*
