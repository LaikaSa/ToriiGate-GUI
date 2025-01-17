import sys
import os
from PySide6.QtWidgets import QApplication
from gui import MainWindow

if __name__ == "__main__":
    # Get quantization mode from environment variable
    quant_mode = os.environ.get('QUANT_MODE', None)
    
    app = QApplication(sys.argv)
    window = MainWindow(quantization_mode=quant_mode)
    window.show()
    sys.exit(app.exec())