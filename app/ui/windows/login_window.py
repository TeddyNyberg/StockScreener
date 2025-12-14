from PySide6.QtWidgets import (QVBoxLayout, QDialog, QLineEdit, QDialogButtonBox, QFrame, QMessageBox, QPushButton)
from app.db.db_handler import authenticate_user, register_user


class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.user_id = None

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.handle_login)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        self.create_account_btn = QPushButton("Create Account")
        self.create_account_btn.setFlat(True)
        self.create_account_btn.setStyleSheet("""
            QPushButton {
                color: #0066cc;
                text-align: center;
                text-decoration: underline;
                border: none;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #004488;
            }
        """)
        self.create_account_btn.clicked.connect(self.handle_create_account)
        layout.addWidget(self.create_account_btn)
        self.setLayout(layout)


    def get_username(self):
        return self.username_input.text()

    def get_password(self):
        return self.password_input.text()

    def handle_login(self):
        username = self.get_username()
        password = self.get_password()

        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Please enter both a username and a password.")
            return

        user_id = authenticate_user(username, password)

        if user_id:
            self.user_id = user_id
            print(f"Login approved for user ID: {user_id}")
            self.accept()
        else:
            QMessageBox.question(
                self,
                "Login Failed",
                "Invalid username or password",
                QMessageBox.StandardButton.Ok
            )


    def handle_create_account(self):
        username = self.get_username()
        password = self.get_password()

        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Please enter a username and password to create an account.")
            return

        self.attempt_registration(username, password)

    def attempt_registration(self, username, password):
        user_id = register_user(username, password)

        if user_id:
            self.user_id = user_id
            QMessageBox.information(self, "Success", "Account created successfully! Logging you in.")
            print(f"Registration successful for user ID: {user_id}")
            self.accept()
        else:
            QMessageBox.warning(self, "Registration Failed",
                                "Could not create account.\nThe username might already be taken.")