@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing latest dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Verifying CUDA installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo Setup complete!
pause