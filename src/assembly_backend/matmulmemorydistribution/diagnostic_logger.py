import pandas as pd
import threading

class DiagnosticLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DiagnosticLogger, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.tables = {
            'nodes': [],
            'ops': [],
            'memory': [],
            'events': [],
            'errors': []
        }
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, table, **kwargs):
        if self.enabled:
            self.tables.setdefault(table, []).append(kwargs)

    def to_df(self, table):
        return pd.DataFrame(self.tables.get(table, []))

    def dump_all(self, dirpath):
        import os
        os.makedirs(dirpath, exist_ok=True)
        for table, rows in self.tables.items():
            pd.DataFrame(rows).to_csv(os.path.join(dirpath, f"{table}.csv"), index=False)

# Singleton instance
logger = DiagnosticLogger()
