import pickle
import os
import random


def load_pkl_file(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"❌ 文件不存在: {pkl_path}")
        return []

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def show_data(data_list, pkl_path, only_errors=False, random_mode=False):
    print(f"\n📄 当前数据文件路径：{pkl_path}")

    if only_errors:
        data_list = [item for item in data_list if item.get("answerKey") != item.get("model_ans")]

    if not data_list:
        print("⚠️ 没有可显示的数据（可能全部预测正确）。")
        return

    indices = list(range(len(data_list)))
    if random_mode:
        random.shuffle(indices)

    for i, idx in enumerate(indices, 1):
        item = data_list[idx]
        gold = item.get("answerKey", "")
        pred = item.get("model_ans", "")

        print(f"\n--- 样本 {idx + 1} ---")
        print(f"问题 ID：{item.get('id', 'N/A')}")
        print(f"题干：{item.get('q_content', '（已删除）')}")
        print(f"选项：{item.get('choices', 'N/A')}")
        print(f"正确答案：{gold}")
        print(f"模型答案：{pred}")
        print(f"是否正确：{'✅ 正确' if gold == pred else '❌ 错误'}")

        cmd = input("按 Enter 查看下一条，输入 q 退出：").strip().lower()
        if cmd == "q":
            break


def main():
    # 修改此路径以查看不同模型或协议的结果
    # pkl_path = "output/Qwen2-7B-Instruct/CommonSense_results_2000_Trust.pkl"
    pkl_path = "output\Qwen2-7B-Instruct\CommonSense_results_5000_Correct_Guidance.pkl"
    data_list = load_pkl_file(pkl_path)

    if not data_list:
        return

    while True:
        print("\n请选择操作：")
        print("1. 按顺序查看所有数据")
        print("2. 查看模型预测错误的数据")
        print("3. 随机查看数据")
        print("0. 退出")
        choice = input("请输入选项编号：").strip()

        if choice == "1":
            show_data(data_list, pkl_path, only_errors=False, random_mode=False)
        elif choice == "2":
            show_data(data_list, pkl_path, only_errors=True, random_mode=False)
        elif choice == "3":
            show_data(data_list, pkl_path, only_errors=False, random_mode=True)
        elif choice == "0":
            print("退出程序。")
            break
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()
