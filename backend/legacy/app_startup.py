from backend.app.db.db_handler import init_db, init_user_table
from backend.app.db.database import DB


def start_application():
    from backend.legacy.ui.windows.main_window import MainWindow
    print("Starting app...")
    #run_backtesting()
    with DB() as conn:
        init_user_table(conn)
        init_db(conn)
    window = MainWindow()
    window.show()
    return window