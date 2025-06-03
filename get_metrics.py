import json
import re
import os
from pipeline import data_length, method


def load_json_results(file_path):
    """
    加载实验结果的JSON文件。
    Args:
        file_path (str): JSON文件的路径。
    Returns:
        list: 包含实验结果字典的列表，如果文件不存在则返回空列表。
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics(results_data, protocol_type, raw_results_data=None):
    """
    计算给定实验结果数据的准确率、从众率和独立率。
    Args:
        results_data (list): 单一协议的实验结果列表。
        protocol_type (str): 协议类型 (例如 "Raw", "Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt")。
        raw_results_data (list, optional): Raw 协议的实验结果列表，用于计算从众率和独立率。
    Returns:
        dict: 包含准确率、从众率和独立率的字典。
    """
    total_questions = len(results_data)
    correct_predictions = 0

    if total_questions == 0:
        return {
            "protocol_type": protocol_type,
            "total_questions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "conformity_rate": 0.0,  # 初始值，如果协议不计算则保持0.0或N/A
            "independence_rate": 0.0  # 初始值
        }

    # 计算准确率
    for item in results_data:
        model_final_ans_label = item.get("model_ans", "N/A")
        gold_ans_label = item.get("correct_ans", "N/A")
        if model_final_ans_label == gold_ans_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0.0

    metrics = {
        "protocol_type": protocol_type,
        "total_questions": total_questions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "conformity_rate": "N/A",  # 默认为N/A，只在特定协议下计算
        "independence_rate": "N/A"  # 默认为N/A，只在特定协议下计算
    }

    # 计算从众率 (Conformity Rate)
    if protocol_type in ["Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt"] and raw_results_data:
        raw_correct_questions_count = 0
        raw_wrong_questions_count = 0
        raw_results_map = {item["id"]: item for item in raw_results_data}

        # 统计 Raw 策略下正确和错误的问题数
        for item in raw_results_data:
            if item.get("model_ans") == item.get("correct_ans"):
                raw_correct_questions_count += 1
            else:
                raw_wrong_questions_count += 1

        conformity_numerator = 0

        for current_item in results_data:
            q_id = current_item.get("id")
            raw_item = raw_results_map.get(q_id)

            if not raw_item:
                continue

            current_model_ans = current_item.get("model_ans", "N/A")
            gold_ans = current_item.get("correct_ans", "N/A")
            raw_model_ans = raw_item.get("model_ans", "N/A")

            # 确定多数意见
            majority_opinion_label = None
            choices = current_item.get("choices", [])
            answer_key = current_item.get("answerKey", "")

            if protocol_type in ["Correct_Guidance", "Doubt"]:
                majority_opinion_label = answer_key  # 多数意见是正确答案
            elif protocol_type in ["Wrong_Guidance", "Trust"]:
                incorrect_choices = [c['label'] for c in choices if c['label'] != answer_key]
                if incorrect_choices:
                    # 假设其他代理给出了一个预设的错误答案，这里为了计算我们取第一个不正确的作为代表
                    majority_opinion_label = incorrect_choices[0]
                else:
                    # 如果没有错误选项，这种情况不应发生在设计为有错误答案的协议中
                    majority_opinion_label = answer_key  # 理论上不应该到这里

            if majority_opinion_label is None or current_model_ans == 'N/A' or raw_model_ans == 'N/A':
                continue  # 数据不完整，跳过此问题

            # 从众率计算逻辑
            if protocol_type in ["Wrong_Guidance", "Trust"]:  # 多数意见是错误的
                # 如果 Raw 策略时回答正确 并且 当前策略下回答错误 (即从众了错误的多数意见)
                if raw_model_ans == gold_ans and current_model_ans != gold_ans:
                    conformity_numerator += 1

                if raw_correct_questions_count > 0:
                    metrics["conformity_rate"] = (conformity_numerator / raw_correct_questions_count) * 100
                else:
                    metrics["conformity_rate"] = 0.0  # 避免除以零
            elif protocol_type in ["Correct_Guidance", "Doubt"]:  # 多数意见是正确的
                # 如果 Raw 策略时回答错误 并且 当前策略下回答正确 (即从众了正确的多数意见)
                if raw_model_ans != gold_ans and current_model_ans == gold_ans:
                    conformity_numerator += 1

                if raw_wrong_questions_count > 0:
                    metrics["conformity_rate"] = (conformity_numerator / raw_wrong_questions_count) * 100
                else:
                    metrics["conformity_rate"] = 0.0  # 避免除以零

    return metrics


def calculate_independence_rate(raw_results_data, trust_results_data, doubt_results_data):
    """
    计算独立率。
    独立率等于 在trust策略、doubt策略、raw策略时都回答正确的问题个数 / 在raw策略下回答正确的问题个数
    Args:
        raw_results_data (list): Raw 协议的实验结果列表。
        trust_results_data (list): Trust 协议的实验结果列表。
        doubt_results_data (list): Doubt 协议的实验结果列表。
    Returns:
        float: 独立率。
    """
    if not raw_results_data:
        return 0.0

    raw_results_map = {item["id"]: item for item in raw_results_data}
    trust_results_map = {item["id"]: item for item in trust_results_data}
    doubt_results_map = {item["id"]: item for item in doubt_results_data}

    correct_in_raw = 0
    correct_in_all_three_protocols = 0

    for q_id, raw_item in raw_results_map.items():
        raw_model_ans = raw_item.get("model_ans")
        gold_ans = raw_item.get("correct_ans")

        if raw_model_ans == gold_ans:
            correct_in_raw += 1

            # 检查在 Trust 和 Doubt 协议下是否也回答正确
            trust_item = trust_results_map.get(q_id)
            doubt_item = doubt_results_map.get(q_id)

            if (trust_item and trust_item.get("model_ans") == gold_ans) and \
                    (doubt_item and doubt_item.get("model_ans") == gold_ans):
                correct_in_all_three_protocols += 1

    if correct_in_raw > 0:
        return (correct_in_all_three_protocols / correct_in_raw) * 100
    return 0.0


# 主执行部分
if __name__ == "__main__":
    base_results_dir = "output"  # 你的根结果文件夹

    # 定义要分析的协议类型
    protocols_to_analyze = ["Raw", "Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt"]

    all_models_metrics = {}

    if not os.path.exists(base_results_dir):
        print(f"错误: 结果目录 '{base_results_dir}' 不存在。请检查路径。")
    else:
        for model_name in os.listdir(base_results_dir):
            model_dir = os.path.join(base_results_dir, model_name)
            if os.path.isdir(model_dir):
                print(f"\n--- 正在处理模型: {model_name} ---")
                current_model_metrics = {}

                if model_name == "DeepSeek-R1-Distill-Qwen-14B" and method == "self-consistency":
                    data_length_here = 1000
                else:
                    data_length_here = data_length


                # 首先加载 Raw 协议的结果，以便在其他协议计算从众率时使用
                raw_file_name = f"CommonSense_results_{data_length}_Raw.json"
                raw_results_file_path = os.path.join(model_dir, raw_file_name)
                raw_results_data = load_json_results(raw_results_file_path)

                if not raw_results_data:
                    print(f"警告: 未能加载模型 {model_name} 的 Raw 协议数据，可能无法计算从众率和独立率。")

                # 加载 Trust 和 Doubt 协议的结果，以便计算独立率
                trust_file_name = f"CommonSense_results_{data_length_here}{method}_Trust.json"
                trust_results_file_path = os.path.join(model_dir, trust_file_name)
                trust_results_data = load_json_results(trust_results_file_path)

                doubt_file_name = f"CommonSense_results_{data_length_here}{method}_Doubt.json"
                doubt_results_file_path = os.path.join(model_dir, doubt_file_name)
                doubt_results_data = load_json_results(doubt_results_file_path)

                for protocol in protocols_to_analyze:
                    if protocol == "Raw":
                        file_name = f"CommonSense_results_{data_length}_{protocol}.json"
                    else:
                        file_name = f"CommonSense_results_{data_length_here}{method}_{protocol}.json"
                    results_file_path = os.path.join(model_dir, file_name)

                    print(f"  - 正在分析 {protocol} 协议...")
                    results_data = load_json_results(results_file_path)

                    if results_data:
                        metrics = calculate_metrics(results_data, protocol, raw_results_data)
                        current_model_metrics[protocol] = metrics

                        print(f"    总问题数: {metrics['total_questions']}")
                        print(f"    准确率: {metrics['accuracy']:.2f}%")
                        if metrics['conformity_rate'] != "N/A":
                            print(f"    从众率: {metrics['conformity_rate']:.2f}%")
                    else:
                        print(f"    未能加载 {protocol} 协议的数据 ({results_file_path})，跳过计算。")

                # 在所有协议的指标都计算完毕后，统一计算独立率
                if raw_results_data and trust_results_data and doubt_results_data:
                    independence_rate_val = calculate_independence_rate(raw_results_data, trust_results_data,
                                                                        doubt_results_data)
                    # 将独立率添加到 Trust 或 Doubt 协议的 metrics 中，或者作为单独的指标
                    # 这里选择添加到 Trust 协议的 metrics 中，因为它是长期策略的一部分
                    if "Trust" in current_model_metrics:
                        current_model_metrics["Trust"]["independence_rate"] = independence_rate_val
                    if "Doubt" in current_model_metrics:  # 也可以选择添加到 Doubt
                        current_model_metrics["Doubt"]["independence_rate"] = independence_rate_val
                    print(f"    独立率 (适用于 Trust/Doubt): {independence_rate_val:.2f}%")
                else:
                    print("    未能加载所有必要数据 (Raw, Trust, Doubt) 以计算独立率。")

                all_models_metrics[model_name] = current_model_metrics

    print("\n--- 所有模型和协议的指标概览 ---")
    for model, protocols_metrics in all_models_metrics.items():
        print(f"\n模型: {model}")
        for protocol, metrics in protocols_metrics.items():
            print(f"  [{protocol}]")
            print(f"    准确率: {metrics['accuracy']:.2f}%")
            if metrics['conformity_rate'] != "N/A":
                print(f"    从众率: {metrics['conformity_rate']:.2f}%")
            if 'independence_rate' in metrics and metrics['independence_rate'] != "N/A":
                print(f"    独立率: {metrics['independence_rate']:.2f}%")

    # 将所有指标保存到一个总的JSON文件中
    output_metrics_file = f"all_models_metrics_summary_{data_length}{method}.json"
    with open(output_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_models_metrics, f, ensure_ascii=False, indent=4)
    print(f"\n所有模型的汇总指标已保存到 {output_metrics_file}")