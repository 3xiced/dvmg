# DVMG

# Install

## Windows

 - Create virtual environment
```
python -m venv env
```
 - Activate virtual environment
```
.\env\Scripts\Activate.ps1
```
 - Install requirements
```
pip install -r requirements.txt
```
 - Run
```
python app
```

## Linux

 - Create virtual environment
```
python3 -m venv env
```
 - Activate virtual environment
```
source env/bin/activate
```
 - Install requirements
```
pip install -r requirements.txt
```
 - Run
```
python app
```

# Usage
## Desktop
При выбирании динамических уравнений, N - массив, i - индексатор, k - коэффициент (=20)
Нельзя использовать индексы > i+k. (e.g. i+2*k, i+3*k/2 etc.)
## Console
В коде меняется метод реконструкции и др. параметры. Файлы сохраняются в ./dataset
