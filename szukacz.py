# program przegladajacy folder w ktorym sue znajduje i wszystko podflodery w poszukiwaniu plikow z rozszerzeniem .py i zapisuje je do pliku .txt w fromacie nazwa kod w danym pliku, pomijajac folder .venv

import os


def find_py_files(root_folder: str, output_file: str):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Pomijamy folder .venv
            if ".venv" in dirnames:
                dirnames.remove(".venv")
            if "legacy" in dirnames:
                dirnames.remove("legacy")
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    out_f.write(f"--- File: {file_path} ---\n")
                    out_f.write(code + "\n\n")


if __name__ == "__main__":
    find_py_files(".", "found_py_files.txt")
