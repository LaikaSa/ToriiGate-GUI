@echo off
setlocal

:: Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip and install build tools
python -m pip install --upgrade pip
pip install setuptools wheel packaging ninja cmake

:: Install PyTorch with CUDA 12.1 first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install other requirements
pip install transformers bitsandbytes accelerate Pillow PySide6 exllamav2

:: Set required environment variables
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
set MAX_JOBS=4

:: Install Flash Attention without custom flags
pip install flash-attn ^
    --no-cache-dir ^
    --no-build-isolation

:: Build ExLlamaV2 kernels
python -c "from exllamav2 import ExLlamaV2"

echo Installation complete!
echo To activate venv later: venv\Scripts\activate.bat
pause