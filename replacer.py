import json
import os

with open("terms_map.json", "r", encoding="utf-8") as f:
    terms = json.load(fp=f)

base_path = "knowledge_base"


def replace_in_file(file_path, replacements):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    for original, new in replacements.items():
        content = content.replace(original, new)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def replace_in_directory(directory_path, replacements):
    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            replace_in_file(file_path, replacements)

replace_in_directory(base_path, terms)
