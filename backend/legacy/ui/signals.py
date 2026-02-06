from PySide6.QtCore import QObject, Signal

class AppSignals(QObject):
    trade_completed = Signal()

global_signals = AppSignals()