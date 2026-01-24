import pytest
from PySide6.QtWidgets import QApplication
from backend.app.db.db_handler import register_user
from tests.utils import force_delete_user


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_user():
    username = "fixture_user"
    password = "FixturePassword123!"

    force_delete_user(username)
    user_id = register_user(username, password)

    yield username, password, user_id

    force_delete_user(username)


