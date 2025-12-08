from PyQt6.QtWidgets import QVBoxLayout, QDialog, QLineEdit, QDialogButtonBox

from app.db.db_handler import authenticate_user


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

        user_id = authenticate_user(username, password)

        if user_id:
            print("Login approved!")
            self.accept()
        else:
            print("Login denied.")