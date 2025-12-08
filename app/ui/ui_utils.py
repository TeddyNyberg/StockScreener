from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy, QTableWidget, QTableWidgetItem
import pandas as pd

def clear_layout(layout):
    while layout.count() > 0:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()


def make_buttons(button_map, layout):
    button_layout = QHBoxLayout()
    for label, (func, get_args_func) in button_map.items():
        btn = QPushButton(label)
        btn.clicked.connect(lambda _, f=func, a_func=get_args_func: f(*a_func()))
        button_layout.addWidget(btn)
    spacer = QSpacerItem(40, 20, QSizePolicy.Expanding)
    button_layout.addItem(spacer)
    layout.addLayout(button_layout)

def create_table_widget_from_data(info=None, df=None):
    if df is None:
        df = pd.DataFrame(info.items(), columns=["Metric", "Value"])
    table_widget = QTableWidget()
    table_widget.setRowCount(df.shape[0])
    table_widget.setColumnCount(df.shape[1])
    if info is None:
        table_widget.setHorizontalHeaderLabels([str(col.date()) for col in df.columns])
        table_widget.setVerticalHeaderLabels(df.index)
    for row_index in range(df.shape[0]):
        for col_index in range(df.shape[1]):
            value = df.iloc[row_index, col_index]
            table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))
    return table_widget