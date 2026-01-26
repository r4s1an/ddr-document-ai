import hashlib
import streamlit as st
from sqlalchemy import create_engine
from pathlib import Path
import shutil

class AppServices:
    def __init__(self):
        self._engine = None

    def get_engine(self):
        if self._engine is None:
            self._engine = create_engine(
                st.secrets["DATABASE_URL"],
                pool_pre_ping=True
            )
        return self._engine

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

def clear_dir_contents(dir_path: Path):
    """Delete all files/folders inside dir_path but keep the directory itself."""
    if not dir_path.exists():
        return
    for p in dir_path.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
