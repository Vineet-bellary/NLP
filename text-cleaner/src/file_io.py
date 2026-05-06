import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def read_file(file_name) -> str:
    file_path = os.path.join(DATA_DIR, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read(), file_path
    except FileNotFoundError as e:
        print(f"file {file_path} not found:\n {e}")
        return None, file_path
