# ToriiGate-GUI
Simple interface for ToriiGate Image caption model
<br/> Use default v0.3, v0.4 models:
<br/> run setup.bat
<br/> choose quantization depend on your available VRAM (default is doable with 24GB VRAM NVIDIA GPU)
<br/> the GUI let you choose 3 optional user prompts recommended by developers, more info: https://rentry.co/q4pisesb
<br/> 
<br/> Use exllamav2 models: 
<br/> Switch to exllamav2 branch
<br/> prepare tools to build flash attention: Visual Studio 2022 with:"Desktop development with C++" workload, Windows 10/11 SDK, CMake tools
<br/> run Setup.bat and wait for it to build flash attention, gonna take a while so go grab a coffe or something
<br/> manually download the ToriiGate-v0.4-7B-exl2-#bpw models on huggingface and put them in the models folder in the root folder of this repo
<br/> launch_default.bat
