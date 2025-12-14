from app.ui.windows.login_window import LoginWindow

def test_login_inputs_exist(qtbot):
    window = LoginWindow()
    qtbot.addWidget(window)

    qtbot.keyClicks(window.username_input, "testuser")
    qtbot.keyClicks(window.password_input, "password123")

    assert window.username_input.text() == "testuser"
    assert window.password_input.text() == "password123"

def test_create_account_link(qtbot):
    window = LoginWindow()
    qtbot.addWidget(window)
    assert "Create Account" in window.create_account_btn.text()