# 🚖 RideShare Prediction Model

This project demonstrates a supervised learning workflow using a rideshare dataset. The script performs data loading, preprocessing, model training, and evaluation. Additionally, it integrates with New Relic for performance monitoring.

## 📑 Table of Contents

- [📥 Installation](#installation)
- [🚀 Usage](#usage)
- [⚙️ Configuration](#configuration)
- [📦 Dependencies](#dependencies)
- [📜 License](#license)

## 📥 Installation

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Install additional dependencies for Parquet support**:
    ```sh
    pip install pyarrow  # or fastparquet
    ```

## 🚀 Usage

1. **Configure New Relic**:
    - Ensure you have a `newrelic.ini` file with the appropriate configuration, including the `license_key`.

2. **Run the script**:
    ```sh
    python supervised_learning.py
    ```

## ⚙️ Configuration

- **New Relic Configuration**:
    - The script initializes the New Relic agent using the `newrelic.ini` file. Ensure this file is present in the project directory and contains the necessary configuration.

## 📦 Dependencies

- `pandas`
- `scikit-learn`
- `newrelic`
- `ml_performance_monitoring`
- `pyarrow` or `fastparquet` (for Parquet file support)

Install all dependencies using:
```sh
pip install -r requirements.txt
```

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.