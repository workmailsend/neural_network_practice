"""Настройки читаются из переменных окружения (.env через python-dotenv)
и значений по умолчанию - никаких захардкоженных путей в коде."""
import os

from dotenv import load_dotenv

# подхватываем .env из текущей папки, если он есть - без него просто
# используются системные переменные окружения или значения по умолчанию
load_dotenv()

PUT_PO_UMOLCHANIU = os.getenv("PREPARE_DATA_PATH", "../data/input")
VYHOD_PO_UMOLCHANIU = os.getenv("PREPARE_OUTPUT_DIR", "../data/output")
UROVEN_LOGOV = os.getenv("PREPARE_LOG_LEVEL", "INFO")
