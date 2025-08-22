from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from app.ui import MainWindow

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()