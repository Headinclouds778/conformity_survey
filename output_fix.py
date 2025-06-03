import json
import re
import os

def extract_model_ans(protocol_result):
    """
    更宽松地从 protocol_result 中提取模型答案的 label
    """
    if not isinstance(protocol_result, str):
        return ""
    match = re.search(r'[\("]?([A-E])[\)\.][ ]', protocol_result)
    if match:
        return match.group(1)
    return ""

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count_modified = 0

    for item in data:
        if item.get("model_ans", "") == "":
            protocol_result = item.get("protocol_result", "")
            extracted = extract_model_ans(protocol_result)
            if extracted:
                item["model_ans"] = extracted
                count_modified += 1

    if count_modified > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"{file_path}: 补全了 {count_modified} 个 model_ans。")
    else:
        print(f"{file_path}: 所有 model_ans 已经存在，无需修改。")

if __name__ == "__main__":
    root_dir = "output"
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(subdir, filename)
                process_json_file(filepath)
