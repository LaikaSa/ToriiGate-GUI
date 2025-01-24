from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                              QPushButton, QTextEdit, QFileDialog, QRadioButton, QApplication,
                              QButtonGroup, QCheckBox, QProgressBar, QLabel, QLineEdit)
from PySide6.QtCore import Qt, Signal, QThread
import torch
from processor import ImageProcessor
from processor import AVAILABLE_MODELS 

class ProcessingThread(QThread):
    finished = Signal(dict)
    progress = Signal(int)
    
    def __init__(self, processor, mode, image_path, prompt_type, use_tags, prefix=""):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.image_path = image_path
        self.prompt_type = prompt_type
        self.use_tags = use_tags
        self.prefix = prefix
        
    def run(self):
        if self.mode == "single":
            result = self.processor.process_single_image(
                self.image_path, 
                self.prompt_type,
                self.use_tags
            )
            self.finished.emit({"result": result})
        else:
            results = self.processor.process_batch(
                self.image_path,
                prompt_type=self.prompt_type,
                use_tags=self.use_tags,
                prefix=self.prefix
            )
            self.finished.emit(results)

class MainWindow(QMainWindow):
    def __init__(self, quantization_mode=None):
        super().__init__()
        self.quantization_mode = quantization_mode
        self.processor = ImageProcessor(list(AVAILABLE_MODELS.keys())[0], quantization_mode)
        self.setup_ui()
        self.update_status_bar()
        
    def update_status_bar(self):
        status_msg = f"Model: {self.model_combo.currentText()} | Device: {self.processor.device} "
        if torch.cuda.is_available():
            status_msg += f"({torch.cuda.get_device_name(0)}) "
        status_msg += f"| Quantization: {self.quantization_mode if self.quantization_mode else 'None'}"
        self.statusBar().showMessage(status_msg)
        
    def setup_ui(self):
        self.setWindowTitle("ToriiGate Image Processor")
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()
        self.single_mode = QRadioButton("Single Image")
        self.batch_mode = QRadioButton("Batch Processing")
        self.mode_group.addButton(self.single_mode)
        self.mode_group.addButton(self.batch_mode)
        self.single_mode.setChecked(True)
        mode_layout.addWidget(self.single_mode)
        mode_layout.addWidget(self.batch_mode)
        layout.addLayout(mode_layout)
        
        # Prompt type selection
        prompt_layout = QHBoxLayout()
        self.prompt_group = QButtonGroup()
        self.json_prompt = QRadioButton("JSON Format")
        self.detailed_prompt = QRadioButton("Long Description")
        self.brief_prompt = QRadioButton("Brief Description")
        self.prompt_group.addButton(self.json_prompt)
        self.prompt_group.addButton(self.detailed_prompt)
        self.prompt_group.addButton(self.brief_prompt)
        self.json_prompt.setChecked(True)
        prompt_layout.addWidget(self.json_prompt)
        prompt_layout.addWidget(self.detailed_prompt)
        prompt_layout.addWidget(self.brief_prompt)
        layout.addLayout(prompt_layout)
        
        # Tags option
        self.use_tags = QCheckBox("Use Tags (if available)")
        layout.addWidget(self.use_tags)

        # Add prefix option
        prefix_layout = QHBoxLayout()
        self.prefix_label = QLabel("Caption Prefix:")
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Optional: Add prefix to all captions")
        prefix_layout.addWidget(self.prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        layout.addLayout(prefix_layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        self.select_button = QPushButton("Select File/Folder")
        file_layout.addWidget(self.path_label)
        file_layout.addWidget(self.select_button)
        layout.addLayout(file_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        # Process button
        self.process_button = QPushButton("Process")
        layout.addWidget(self.process_button)
        
        # Connect signals
        self.select_button.clicked.connect(self.select_path)
        self.process_button.clicked.connect(self.process_images)
        self.single_mode.toggled.connect(self.update_ui)
        # Connect model change signal
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

    def on_model_changed(self, model_name):
        # Disable UI during model switch
        self.setEnabled(False)
        self.statusBar().showMessage("Switching model... Please wait.")
        QApplication.processEvents()  # Update UI

        try:
            # Unload current model
            if hasattr(self, 'processor'):
                print("Unloading current model...")
                self.processor.model = None
                self.processor.processor = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory cleared. Current usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # Initialize new model
            print(f"Loading new model: {model_name}")
            self.processor = ImageProcessor(model_name, self.quantization_mode)
            self.update_status_bar()
            print("Model switch completed successfully")

        except Exception as e:
            print(f"Error switching models: {str(e)}")
            self.statusBar().showMessage(f"Error switching models: {str(e)}")

        finally:
            # Re-enable UI
            self.setEnabled(True)
        
    def select_path(self):
        if self.single_mode.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Image",
                filter="Images (*.jpg *.png *.webp *.jpeg)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select Folder"
            )
            
        if path:
            self.path_label.setText(path)
            
    def get_prompt_type(self):
        if self.json_prompt.isChecked():
            return "json"
        elif self.detailed_prompt.isChecked():
            return "long"
        return "short"
        
    def process_images(self):
        path = self.path_label.text()
        if path == "No file selected":
            return

        if "exl2" in self.model_combo.currentText().lower():
            self.use_tags.setText("Use Metadata Files (tags, chars, traits)")
        else:
            self.use_tags.setText("Use Tags (if available)")

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.processing_thread = ProcessingThread(
            self.processor,
            "single" if self.single_mode.isChecked() else "batch",
            path,
            self.get_prompt_type(),
            self.use_tags.isChecked(),
            self.prefix_input.text()  # Add prefix from input
        )
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
        
    def on_processing_finished(self, results):
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if isinstance(results, dict):
            output = ""
            for key, value in results.items():
                output += f"File: {key}\n{value}\n\n"
            self.output_text.setText(output)
            
    def update_ui(self):
        is_single = self.single_mode.isChecked()
        self.select_button.setText("Select File" if is_single else "Select Folder")