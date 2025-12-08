from app.db.db_handler import DB, init_db
from app.ml_logic.tester import run_backtesting


def start_application():
    from app.ui.windows.main_window import MainWindow
    print("Starting app...")
    run_backtesting()
    with DB() as conn:
        init_db(conn)
    window = MainWindow()
    window.show()