from PyQt6.QtWidgets import QVBoxLayout, QDialog, QLineEdit, QDialogButtonBox, QPushButton, QMessageBox

from app.db.db_handler import authenticate_user, register_user


class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        self.create_account_btn = QPushButton("Create Account")
        self.create_account_btn.clicked.connect(self.handle_create_account)
        layout.addWidget(self.create_account_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.handle_login)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
            print(f"Login approved for user ID: {user_id}")
            self.accept()
        else:
            reply = QMessageBox.question(
                self,
                "Login Failed",
                "Invalid credentials.\n\nWould you like to create a new account with this username and password?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.attempt_registration(username, password)

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
            QMessageBox.information(self, "Success", "Account created successfully! Logging you in.")
            print(f"Registration successful for user ID: {user_id}")
            self.accept()
        else:
            QMessageBox.warning(self, "Registration Failed",
                                "Could not create account.\nThe username might already be taken.")