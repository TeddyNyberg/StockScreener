from app.ui.windows.login_window import LoginWindow
from app.db.db_handler import authenticate_user, register_user, DB, init_user_table
from tests.utils import force_delete_user


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


def test_register_user():
    test_username = "mock_user"
    test_password = "SecurePassword123!"

    force_delete_user(test_username)

    user_id = register_user(test_username, test_password)
    assert user_id is not False, "Register user returned False (failed)."

    user_id_v2 = register_user(test_username, test_password)
    assert user_id_v2 is False, "Created account when username was taken (failed)."

    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()

            assert result is not None, "User ID returned by register function but not found in DB."
            assert result[0] == test_username, "Username in DB does not match input."
            assert result[1] != test_password, "SECURITY RISK: Password stored in plain text!"

    force_delete_user(test_username)


def test_authenticate_user(temp_user):
    username, password, user_id = temp_user
    assert authenticate_user(username, password) == user_id
    assert authenticate_user(username, "wrong_pass") is False


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

    # --- 4. Verify User is in Table ---
    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()

            assert result is not None, "User ID returned by register function but not found in DB."
            assert result[0] == test_username, "Username in DB does not match input."
            assert result[1] != test_password, "SECURITY RISK: Password stored in plain text!"

    # --- 5. Check Login (Authenticate) ---
    # Should succeed with correct password
    login_id = authenticate_user(test_username, test_password)
    assert login_id == user_id, "Login failed with correct password."

    # Should fail with wrong password
    bad_login = authenticate_user(test_username, "WrongPass")
    assert bad_login is False, "Login succeeded with WRONG password."

    # --- 6. Delete Mock User (Cleanup) ---
    force_delete_user(test_username)

    # Verify deletion by trying to login again
    deleted_login = authenticate_user(test_username, test_password)
    assert deleted_login is False, "User should be deleted but login still works."



