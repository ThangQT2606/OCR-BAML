# Invoices_LLM

BAML + LLM

## 1. Install Necessary Libraries

```bash
    conda create -n ocr_baml python=3.10.12
    pip install -r requirements.txt
```

## 2. Generator Baml

```bash
    baml-cli generate
```

## 3. Run Programming

Create .env file and write GOOGLE_API_KEY="Your_api_key" or export GOOGLE_API_KEY = "Your_api_key"

### With Windows:

```bash
    python engine.py
```

### With MacOS/Linux:

#### 1. Run Python

```bash
    python engine.py
```

#### 2. Shell

```bash
    bash .\run.sh 
```

## 4. Result

Input: 

<img src="images/3.png">

Output:

```
[{'file_name': '3.png', 'extract_data': {'datetime': '07/10/2019', 'products': {'product_name': ['Suon bo ham', 'Banh my ca moi pate', 'Bo thap cam hap', 'Banh bot dau xanh ran thit', 'Cafe nau da', 'Cocacola', 'Banh my ca moi thit nguoi'], 'product_quantity': ['2', '1', '1', '1', '1', 
'1', '1'], 'product_unit_price': ['60,000.00', '45,000.00', '50,000.00', '40,000.00', '32,000.00', '10,000.00', '45,000.00'], 'product_price': ['120,000.00', '45,000.00', '50,000.00', '40,000.00', '32,000.00', '10,000.00', '45,000.00']}, 'total': '342,000.00'}, 'tokens': [1975, 398]}]
```