from backend.app.db.db_handler import DB, init_db, init_user_table
from backend.app.ml_logic.tester import run_backtesting


def start_application():
    from backend.app.ui.windows.main_window import MainWindow
    print("Starting app...")
    run_backtesting()
    with DB() as conn:
        init_user_table(conn)
        init_db(conn)
    window = MainWindow()
    window.show()
    return window