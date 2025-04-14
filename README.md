
# BearingWatch – Intelligent Bearing Fault Detection

[![Website](https://img.shields.io/badge/Live%20App-bearingwatch.com-blue)](https://bearingwatch.com/)
[![Portfolio](https://img.shields.io/badge/Author-mhrafiei.github.io-lightgrey)](https://mhrafiei.github.io/)

BearingWatch is a web-based diagnostic tool designed to detect **rolling element bearing faults** using uploaded vibration signal data. Powered by advanced signal processing techniques, the tool identifies and visualizes key defect frequencies to support predictive maintenance and reduce unplanned downtime.

> 🚀 Try it live at: [https://bearingwatch.com/](https://bearingwatch.com/)

---

## 🔧 What It Does

BearingWatch processes time-series signals from rotating machinery and detects common bearing fault signatures using frequency analysis. The app identifies characteristic frequencies such as:

- **Ball Pass Frequency (Inner Race)**  
- **Ball Pass Frequency (Outer Race)**  
- **Fundamental Train Frequency**  
- **Ball Spin Frequency**  

These features are extracted using advanced signal processing and machine learning methods, and the output is a downloadable diagnostic report to assist in condition monitoring workflows.

---

## 📂 How to Use

1. **Upload your signal**: Provide a `.csv` file with a single column of time-domain signal data.
2. **Enter key parameters**: Input machine-specific variables (e.g., shaft speed, bearing geometry).
3. **Submit Data**: The app performs feature extraction, fault frequency analysis, and returns a detailed report in seconds.

> ⚠️ Make sure your CSV file has **only one column** of numerical data.

---

## 🔐 API Key Configuration

To run the app locally, you'll need to set an environment variable for the OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## 🛠️ Project Structure

```text
├── app.py             # Main Flask application
├── index.html         # Landing page template
├── templates/         # HTML templates
├── static/            # Static CSS/JS files
├── report/            # Example of generated plots (reports are generated in the app)
├── utilities.py       # Signal processing and frequency analysis
├── fetchpull.sh       # Git helper script
├── gitcommit.sh       # Git helper script
├── requirements.txt   # Required libraries
└── README.md          # This file
```

---

## 🚀 Running Locally

Install dependencies and start the Flask server:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python app.py
```

Then open `http://127.0.0.1:8080` in your browser.

---

## 📈 Example Use Case

- An engineer uploads vibration data from a monitored motor.
- Enters bearing specifications and RPM.
- Receives a PDF report showing likely fault types based on spectral patterns.
- Uses the output to schedule targeted maintenance.

---

## 🌐 About

This project is developed and maintained by [Mohammad H. Rafiei](https://mhrafiei.github.io/), a researcher and engineer focused on predictive maintenance and intelligent monitoring solutions.

🔗 Visit my homepage: [https://mhrafiei.github.io](https://mhrafiei.github.io)

---

## 📄 License

This project is licensed under the MIT License.

---

*“Signal intelligence meets reliability.”*  
*Built for the next generation of industrial diagnostics.*

---

