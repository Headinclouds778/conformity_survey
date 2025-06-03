# import json
# import matplotlib
#
# # 尝试设置后端为 'TkAgg'。这需要在导入 matplotlib.pyplot 之前设置。
# # 确保你的Python环境安装了tkinter库。
# try:
#     matplotlib.use('TkAgg')
# except ImportError:
#     print("警告: TkAgg 后端需要安装 tkinter。尝试其他后端或安装 tkinter。")
#     print("如果问题仍然存在，请尝试在 PyCharm 设置中禁用 'Show plots in tool window'。")
#     # 如果 TkAgg 失败，可以回退到 Agg，但这样就不会显示交互式窗口
#     matplotlib.use('Agg')
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # --- Matplotlib 中文字体设置 ---
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
#
# # --- 解决 MatplotlibDeprecationWarning ---
# method_names_list = ["Raw", "Reflection", "Role", "Self-Consistency"]
# num_methods = len(method_names_list)
# colors = [plt.cm.Dark2(i) for i in range(num_methods)]
#
# # 从文件中加载数据
# try:
#     with open('all_models_metrics_summary_5000.json', 'r', encoding='utf-8') as f:
#         data_raw = json.load(f)
#     with open('all_models_metrics_summary_5000reflection.json', 'r', encoding='utf-8') as f:
#         data_reflection = json.load(f)
#     with open('all_models_metrics_summary_5000role.json', 'r', encoding='utf-8') as f:
#         data_role = json.load(f)
#     with open('all_models_metrics_summary_5000self-consistency.json', 'r', encoding='utf-8') as f:
#         data_self_consistency = json.load(f)
# except FileNotFoundError as e:
#     print(f"错误：文件未找到。请确保所有JSON文件都在脚本运行的目录下。缺失文件：{e.filename}")
#     exit()
#
# # 将所有数据组织起来
# all_metrics_data = {
#     "Raw": data_raw,
#     "Reflection": data_reflection,
#     "Role": data_role,
#     "Self-Consistency": data_self_consistency
# }
#
# # 目标指标
# metrics_to_plot = ["accuracy", "conformity_rate", "independence_rate"]
#
# # 要比较的协议类型 (现在每个协议都会有自己的图)
# protocol_types_to_compare = ["Raw", "Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt"]
#
# # 获取所有模型名称
# model_names = list(data_raw.keys())
#
# # 遍历每个协议类型，生成图表
# for protocol_type in protocol_types_to_compare:
#     # 对于每个协议，我们将有多个子图，每个子图代表一个模型
#     # 总共有 len(model_names) 个子图
#     fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 5 * len(model_names)), sharex=True)
#
#     # 确保 axes 是一个数组，即使只有一个模型
#     if len(model_names) == 1:
#         axes = [axes]  # 将单个轴对象包装在列表中
#
#     fig.suptitle(f'协议: {protocol_type} - 各模型在不同方法下的表现', fontsize=16)
#
#     bar_width = 0.18  # 调整柱子宽度
#     index_offset = np.arange(num_methods) * bar_width  # 计算每种方法的柱子偏移量
#
#     x_positions = np.arange(len(metrics_to_plot))  # 每个指标的位置
#
#     # 遍历每个模型，为当前协议类型下的该模型生成一个子图
#     for model_idx, model_name in enumerate(model_names):
#         ax = axes[model_idx]
#
#         ax.set_title(f'模型: {model_name}', fontsize=12)  # 子图标题显示模型名称
#
#         for method_idx, method_name in enumerate(method_names_list):
#             method_data = all_metrics_data[method_name]
#
#             # 提取当前模型和协议类型下三个指标的值
#             values = []
#             for metric in metrics_to_plot:
#                 val = method_data.get(model_name, {}).get(protocol_type, {}).get(metric)
#                 if val == "N/A" or val is None:
#                     values.append(0)
#                 else:
#                     values.append(val)
#
#             # 绘制柱子
#             rects = ax.bar(x_positions + index_offset[method_idx], values, bar_width,
#                            label=method_name, color=colors[method_idx])
#
#             # 在柱子上显示数值
#             for rect in rects:
#                 height = rect.get_height()
#                 if height != 0:
#                     ax.annotate(f'{height:.2f}',
#                                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                                 xytext=(0, 3),
#                                 textcoords="offset points",
#                                 ha='center', va='bottom', fontsize=7, rotation=45)
#
#         ax.set_ylabel('值 (%)', fontsize=10)  # 纵坐标统一为百分比值
#         ax.set_xticks(x_positions + index_offset.mean())  # 将刻度放在一组柱子的中间
#         ax.set_xticklabels([m.replace("_", " ").capitalize() for m in metrics_to_plot],
#                            rotation=45, ha="right", fontsize=9)
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#         ax.set_ylim(bottom=0)
#
#         # 每个子图都添加图例
#         ax.legend(title="方法", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
#
#     plt.tight_layout(rect=[0, 0.03, 0.95, 0.96])  # 调整布局
#     plt.show()

# import json
# import matplotlib
#
# # 尝试设置后端为 'TkAgg'。这需要在导入 matplotlib.pyplot 之前设置。
# try:
#     matplotlib.use('TkAgg')
# except ImportError:
#     print("警告: TkAgg 后端需要安装 tkinter。尝试其他后端或安装 tkinter。")
#     print("如果问题仍然存在，请尝试在 PyCharm 设置中禁用 'Show plots in tool window'。")
#     # 如果 TkAgg 失败，可以回退到 Agg，但这样就不会显示交互式窗口
#     matplotlib.use('Agg')  # 在非交互式模式下运行，图表会保存到文件
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # --- Matplotlib 中文字体设置 ---
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
#
# # --- 解决 MatplotlibDeprecationWarning ---
# method_names_list = ["Raw", "Reflection", "Role", "Self-Consistency"]
# num_methods = len(method_names_list)
# colors = [plt.cm.Dark2(i) for i in range(num_methods)]
#
# # 从文件中加载数据
# try:
#     with open('all_models_metrics_summary_5000.json', 'r', encoding='utf-8') as f:
#         data_raw = json.load(f)
#     with open('all_models_metrics_summary_5000reflection.json', 'r', encoding='utf-8') as f:
#         data_reflection = json.load(f)
#     with open('all_models_metrics_summary_5000role.json', 'r', encoding='utf-8') as f:
#         data_role = json.load(f)
#     with open('all_models_metrics_summary_5000self-consistency.json', 'r', encoding='utf-8') as f:
#         data_self_consistency = json.load(f)
# except FileNotFoundError as e:
#     print(f"错误：文件未找到。请确保所有JSON文件都在脚本运行的目录下。缺失文件：{e.filename}")
#     exit()
#
# # 将所有数据组织起来
# all_metrics_data = {
#     "Raw": data_raw,
#     "Reflection": data_reflection,
#     "Role": data_role,
#     "Self-Consistency": data_self_consistency
# }
#
# # 目标指标
# metrics_to_plot = ["accuracy", "conformity_rate", "independence_rate"]
#
# # 要比较的协议类型
# protocol_types_to_compare = ["Raw", "Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt"]
#
# # 获取所有模型名称
# model_names = list(data_raw.keys())
#
# # 遍历每个模型和每个协议类型，生成一张图
# for model_name in model_names:
#     for protocol_type in protocol_types_to_compare:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # 每张图一个子图
#
#         fig.suptitle(f'模型: {model_name} - 协议: {protocol_type} - 指标表现', fontsize=16)
#
#         bar_width = 0.18  # 调整柱子宽度
#         index_offset = np.arange(num_methods) * bar_width  # 计算每种方法的柱子偏移量
#
#         x_positions = np.arange(len(metrics_to_plot))  # 每个指标的位置
#
#         for method_idx, method_name in enumerate(method_names_list):
#             method_data = all_metrics_data[method_name]
#
#             # 提取当前模型和协议类型下三个指标的值
#             values = []
#             for metric in metrics_to_plot:
#                 val = method_data.get(model_name, {}).get(protocol_type, {}).get(metric)
#                 if val == "N/A" or val is None:
#                     values.append(0)
#                 else:
#                     values.append(val)
#
#             # 绘制柱子
#             rects = ax.bar(x_positions + index_offset[method_idx], values, bar_width,
#                            label=method_name, color=colors[method_idx])
#
#             # 在柱子上显示数值
#             for rect in rects:
#                 height = rect.get_height()
#                 if height != 0:  # 只有当值不为0时才显示标签
#                     ax.annotate(f'{height:.2f}',
#                                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                                 xytext=(0, 3),  # 3 points vertical offset
#                                 textcoords="offset points",
#                                 ha='center', va='bottom', fontsize=7, rotation=45)
#
#         ax.set_ylabel('值 (%)', fontsize=10)
#         ax.set_xticks(x_positions + index_offset.mean())  # 将刻度放在一组柱子的中间
#         ax.set_xticklabels([m.replace("_", " ").capitalize() for m in metrics_to_plot],
#                            rotation=45, ha="right", fontsize=9)
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#         ax.set_ylim(bottom=0)
#
#         ax.legend(title="方法", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
#
#         plt.tight_layout(rect=[0, 0.03, 0.95, 0.96])  # 调整布局
#         plt.show()
#
#         # 如果你使用 'Agg' 后端（即不显示交互式窗口），可以保存图片
#         # plt.savefig(f'{model_name}_{protocol_type}_metrics.png', dpi=300)
#         # plt.close(fig) # 关闭当前图，释放内存

import json
import matplotlib

# 尝试设置后端为 'TkAgg'。这需要在导入 matplotlib.pyplot 之前设置。
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("警告: TkAgg 后端需要安装 tkinter。尝试其他后端或安装 tkinter。")
    print("如果问题仍然存在，请尝试在 PyCharm 设置中禁用 'Show plots in tool window'。")
    matplotlib.use('Agg')  # 回退到 Agg，图表会保存到文件而不是显示交互式窗口

import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# --- 解决 MatplotlibDeprecationWarning ---
method_names_list = ["Raw", "Reflection", "Role", "Self-Consistency"]
num_methods = len(method_names_list)
colors = [plt.cm.Dark2(i) for i in range(num_methods)]

# 从文件中加载数据
try:
    with open('all_models_metrics_summary_5000.json', 'r', encoding='utf-8') as f:
        data_raw = json.load(f)
    with open('all_models_metrics_summary_5000reflection.json', 'r', encoding='utf-8') as f:
        data_reflection = json.load(f)
    with open('all_models_metrics_summary_5000role.json', 'r', encoding='utf-8') as f:
        data_role = json.load(f)
    with open('all_models_metrics_summary_5000self-consistency.json', 'r', encoding='utf-8') as f:
        data_self_consistency = json.load(f)
except FileNotFoundError as e:
    print(f"错误：文件未找到。请确保所有JSON文件都在脚本运行的目录下。缺失文件：{e.filename}")
    exit()

# 将所有数据组织起来
all_metrics_data = {
    "Raw": data_raw,
    "Reflection": data_reflection,
    "Role": data_role,
    "Self-Consistency": data_self_consistency
}

# 定义所有可能的指标
all_possible_metrics = ["accuracy", "conformity_rate", "independence_rate"]

# 定义不同协议类型对应的指标
protocol_metrics_map = {
    "Correct_Guidance": ["accuracy", "conformity_rate"],
    "Wrong_Guidance": ["accuracy", "conformity_rate"],
    "Trust": ["accuracy", "conformity_rate", "independence_rate"],
    "Doubt": ["accuracy", "conformity_rate", "independence_rate"]
}

# 获取所有模型名称
model_names = list(data_raw.keys())

# 要比较的协议类型 (去除 "Raw")
protocol_types_to_plot = ["Correct_Guidance", "Wrong_Guidance", "Trust", "Doubt"]

# 遍历每个协议类型，生成图表
for protocol_type in protocol_types_to_plot:
    # 获取当前协议应该显示的指标
    current_metrics_to_plot = protocol_metrics_map.get(protocol_type, all_possible_metrics)

    # 对于每个协议，我们将有多个子图，每个子图代表一个模型
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 5 * len(model_names)), sharex=True)

    # 确保 axes 是一个数组，即使只有一个模型
    if len(model_names) == 1:
        axes = [axes]

    if protocol_type != "Correct_Guidance":
        fig.suptitle(f'协议: {protocol_type} - 各模型在不同方法下的表现', fontsize=16)
    else:
        fig.suptitle(f'协议: Right_Guidance - 各模型在不同方法下的表现', fontsize=16)

    bar_width = 0.18  # 调整柱子宽度
    index_offset = np.arange(num_methods) * bar_width  # 计算每种方法的柱子偏移量

    # x轴刻度位置基于当前协议的指标数量
    x_positions = np.arange(len(current_metrics_to_plot))

    # # 遍历每个模型，为当前协议类型下的该模型生成一个子图
    # for model_idx, model_name in enumerate(model_names):
    #     ax = axes[model_idx]
    #
    #     ax.set_title(f'模型: {model_name}', fontsize=12)  # 子图标题显示模型名称
    #
    #     for method_idx, method_name in enumerate(method_names_list):
    #         method_data = all_metrics_data[method_name]
    #
    #         # 提取当前模型和协议类型下三个指标的值
    #         values = []
    #         for metric in current_metrics_to_plot:  # 只遍历当前协议对应的指标
    #             val = method_data.get(model_name, {}).get(protocol_type, {}).get(metric)
    #             if val == "N/A" or val is None:
    #                 values.append(0)
    #             else:
    #                 values.append(val)
    #
    #         # 绘制柱子
    #         rects = ax.bar(x_positions + index_offset[method_idx], values, bar_width,
    #                        label=method_name, color=colors[method_idx])
    #
    #         # 在柱子上显示数值
    #         for rect in rects:
    #             height = rect.get_height()
    #             if height != 0:
    #                 ax.annotate(f'{height:.2f}',
    #                             xy=(rect.get_x() + rect.get_width() / 2, height),
    #                             xytext=(0, 3),
    #                             textcoords="offset points",
    #                             ha='center', va='bottom', fontsize=7, rotation=0)
    #
    #     ax.set_ylabel('值 (%)', fontsize=10)  # 纵坐标统一为百分比值
    #     ax.set_xticks(x_positions + index_offset.mean())  # 将刻度放在一组柱子的中间
    #     # 设置X轴刻度标签为当前协议的指标名称
    #     ax.set_xticklabels([m.replace("_", " ").capitalize() for m in current_metrics_to_plot],
    #                        rotation=45, ha="right", fontsize=9)
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)
    #     ax.set_ylim(bottom=0, top=110)
    #
    #     # 每个子图都添加图例
    #     ax.legend(title="方法", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    # 遍历每个模型，为当前协议类型下的该模型生成一个子图
    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]

        ax.set_title(f'模型: {model_name}', fontsize=12)  # 子图标题显示模型名称

        # --- 新增：存储当前子图所有柱子高度的列表 ---
        current_subplot_bar_heights = []

        for method_idx, method_name in enumerate(method_names_list):
            method_data = all_metrics_data[method_name]

            values = []
            for metric in current_metrics_to_plot:
                val = method_data.get(model_name, {}).get(protocol_type, {}).get(metric)
                if val == "N/A" or val is None:
                    values.append(0)
                else:
                    values.append(val)

            rects = ax.bar(x_positions + index_offset[method_idx], values, bar_width,
                           label=method_name, color=colors[method_idx])

            # 在柱子上显示数值
            for rect in rects:
                height = rect.get_height()
                # --- 新增：将非零的柱子高度添加到列表中 ---
                if height != 0:
                    current_subplot_bar_heights.append(height)

                # ... (ax.annotate 的代码，以及 xytext=(0, x) 的修改) ...
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=0)

        ax.set_ylabel('值 (%)', fontsize=10)
        ax.set_xticks(x_positions + index_offset.mean())
        ax.set_xticklabels([m.replace("_", " ").capitalize() for m in current_metrics_to_plot],
                           rotation=0, ha="right", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # --- 修改这里：动态设置 Y 轴上限 ---
        if current_subplot_bar_heights:  # 确保列表不为空，避免max()出错
            max_height_in_subplot = max(current_subplot_bar_heights)
            ax.set_ylim(bottom=0, top=max_height_in_subplot * 1.1)  # 设置Y轴上限为最大高度的1.2倍
        else:
            ax.set_ylim(bottom=0, top=100)  # 如果没有柱子（数据都为0），给一个默认上限，例如100

        ax.legend(title="方法", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.96])  # 调整布局
    plt.show()