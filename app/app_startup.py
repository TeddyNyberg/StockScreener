from app.db.db_handler import DB, init_db, init_user_table
from app.ml_logic.other_models.interactive_ner_model import interactive_train
from app.ml_logic.tester import run_backtesting


def start_application():
    from app.ui.windows.main_window import MainWindow
    print("Starting app...")
    run_backtesting()
    with DB() as conn:
        init_user_table(conn)
        init_db(conn)
    window = MainWindow()
    window.show()
    interactive_train()
    return window