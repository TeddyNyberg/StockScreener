from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from app.ui import MainWindow

## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

def main():
    print("Starting app...")
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()