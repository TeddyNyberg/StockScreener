from unittest.mock import MagicMock

from PySide6.QtWidgets import QDialog, QMessageBox
from backend.app import LoginWindow
from backend.app.db.db_handler import authenticate_user, register_user, DB, init_user_table
from tests.utils import force_delete_user
import pytest

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

def test_get_username_and_password(qtbot):
    window = LoginWindow()
    qtbot.addWidget(window)
    qtbot.keyClicks(window.username_input, "testuser")
    qtbot.keyClicks(window.password_input, "password123")
    assert window.get_username() == "testuser"
    assert window.get_password() == "password123"


def test_login_success(qtbot, monkeypatch):
    window = LoginWindow()
    qtbot.addWidget(window)

    window.username_input.setText("valid_user")
    window.password_input.setText("correct_pass")

    monkeypatch.setattr("app.ui.windows.login_window.authenticate_user", lambda u, p: 55)

    window.handle_login()

    assert window.user_id == 55
    assert window.result() == QDialog.DialogCode.Accepted


@pytest.mark.parametrize("user_input, pass_input, auth_return", [
    ("bad_user", "wrong_pass", None),
    ("valid_user", "wrong_pass", None),
    ("", "", None),
])
def test_login_failures_parametrized(qtbot, monkeypatch, user_input, pass_input, auth_return):
    window = LoginWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr("app.ui.windows.login_window.authenticate_user", lambda u, p: auth_return)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    window.username_input.setText(user_input)
    window.password_input.setText(pass_input)
    window.handle_login()

    assert window.user_id is None
    assert window.result() != QDialog.DialogCode.Accepted


def test_user_table_exists():
    with DB() as conn:
        init_user_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE  table_schema = 'public'
                    AND    table_name   = 'users'
                );
            """)
            table_exists = cur.fetchone()[0]
            assert table_exists is True, "CRITICAL: 'users' table was not created in the DB."


def test_register_user_success():
    test_username = "mock_user"
    test_password = "SecurePassword123!"

    force_delete_user(test_username)

    user_id = register_user(test_username, test_password)
    assert user_id is not False, "Register user returned False (failed)."

    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()

            assert result is not None, "User ID returned by register function but not found in DB."
            assert result[0] == test_username, "Username in DB does not match input."
            assert result[1] != test_password, "SECURITY RISK: Password stored in plain text!"

    force_delete_user(test_username)


def test_register_duplicate_user_real_db():
    test_username = "duplicate_tester"
    test_password = "password123"

    force_delete_user(test_username)

    first_attempt = register_user(test_username, test_password)
    assert first_attempt is not False, "Setup failed: Could not register user the first time."

    second_attempt = register_user(test_username, test_password)
    assert second_attempt is False, "Database allowed duplicate username!"

    force_delete_user(test_username)

def test_authenticate_user(temp_user):
    username, password, user_id = temp_user
    assert authenticate_user(username, password) == user_id
    assert authenticate_user(username, "wrong_pass") is False


def test_handle_create_account_no_username(qtbot, monkeypatch):
    window = LoginWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    mock_attempt = MagicMock()
    monkeypatch.setattr(window, "attempt_registration", mock_attempt)

    window.handle_create_account()

    mock_attempt.assert_not_called()

def test_handle_create_account_success(qtbot, monkeypatch):
    window = LoginWindow()
    qtbot.addWidget(window)

    window.username_input.setText("valid_user")
    window.password_input.setText("correct_pass")

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    mock_attempt = MagicMock()
    monkeypatch.setattr(window, "attempt_registration", mock_attempt)

    window.handle_create_account()

    mock_attempt.assert_called()


def test_attempt_registration_fail(qtbot, monkeypatch):
    window = LoginWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("app.ui.windows.login_window.register_user", lambda u, p: None)

    window.attempt_registration(None, None)

    assert window.result() != QDialog.DialogCode.Accepted


def test_attempt_registration_success(qtbot, monkeypatch):
    window = LoginWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("app.ui.windows.login_window.register_user", lambda u, p: 55)

    window.attempt_registration(None, None)

    assert window.user_id == 55
    assert window.result() == QDialog.DialogCode.Accepted




def test_user_lifecycle():
    with DB() as conn:
        init_user_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE  table_schema = 'public'
                    AND    table_name   = 'users'
                );
            """)
            table_exists = cur.fetchone()[0]
            assert table_exists is True, "CRITICAL: 'users' table was not created in the DB."

    test_username = "pytest_mock_user"
    test_password = "SecurePassword123!"

    force_delete_user(test_username)

    user_id = register_user(test_username, test_password)
    assert user_id is not False, "Register user returned False (failed)."

    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()

            assert result is not None, "User ID returned by register function but not found in DB."
            assert result[0] == test_username, "Username in DB does not match input."
            assert result[1] != test_password, "SECURITY RISK: Password stored in plain text!"

    login_id = authenticate_user(test_username, test_password)
    assert login_id == user_id, "Login failed with correct password."

    bad_login = authenticate_user(test_username, "WrongPass")
    assert bad_login is False, "Login succeeded with WRONG password."

    force_delete_user(test_username)

    deleted_login = authenticate_user(test_username, test_password)
    assert deleted_login is False, "User should be deleted but login still works."


