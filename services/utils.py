import hashlib
import streamlit as st
from sqlalchemy import create_engine

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