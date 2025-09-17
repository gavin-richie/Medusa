import os
import torch
import numpy as np
from medusa.model.medusa_model import MedusaModel, MedusaConfig
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt
import time


# 设置环境变量
def setup_environment(gpu_id="0", hf_endpoint="https://hf-mirror.com"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["HF_ENDPOINT"] = hf_endpoint
    print(f"环境设置完成: GPU ID={gpu_id}, HF_ENDPOINT={hf_endpoint}")


# 加载模型
def load_model(model_path, medusa_num_heads=4, use_local_base_model=False):
    print(f"开始加载模型: {model_path}")
    start_time = time.time()
    # model_path = 'FasterDecoding/medusa-vicuna-7b-v1.3'

    # 配置参数
    model_kwargs = {
        "medusa_num_heads": medusa_num_heads,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    # 创建Medusa配置对象并设置参数
    # config = MedusaConfig(
    #     medusa_num_heads=4,
    #     medusa_num_layers=1,
    #     base_model_name_or_path=model_path
    # )

    # 如果需要使用本地基础模型路径
    if use_local_base_model:
        model_kwargs["base_model_name_or_path"] = model_path

    # 加载模型
    model = MedusaModel.from_pretrained(
        model_path,
        medusa_num_heads=4,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
        # **model_kwargs
    )

    tokenizer = model.get_tokenizer()
    end_time = time.time()
    print(f"模型加载完成，耗时: {end_time - start_time:.2f}秒")

    return model, tokenizer


# 初始化KV缓存
def init_kv_cache(model):
    print("初始化KV缓存...")
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    print("KV缓存初始化完成")
    return past_key_values, past_key_values_data, current_length_data


# 基本LLM推理示例
def basic_inference_demo(model, tokenizer, prompt):
    print("\n===== 基本LLM推理示例 ======")
    print(f"提示词: {prompt}")

    # 重置KV缓存
    model.current_length_data.zero_()

    # 编码提示词
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    print(f'输入token长度: {input_len}')
    print(f'初始KV缓存形状: {model.past_key_values[0][0].shape} {model.past_key_values[0][1].shape}')

    # 第一次推理（整个提示词）
    with torch.inference_mode():
        output = model.base_model(torch.as_tensor(input_ids).cuda(), past_key_values=model.past_key_values, )
        print(f'输出形状: {output.logits.shape}')
        pred = output.logits.argmax(-1)
        input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
        print(f'第一次推理后KV缓存形状: {model.past_key_values[0][0].shape} {model.past_key_values[0][1].shape}')
        print(f'第一次预测: {tokenizer.batch_decode(pred[..., -1:])}')

    # 第二次推理（使用KV缓存）
    with torch.inference_mode():
        output = model.base_model(pred[..., -1:], past_key_values=model.past_key_values, use_cache=True)
        print(f'输出形状: {output.logits.shape}')
        pred = output.logits.argmax(-1)
        input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
        print(f'第二次推理后KV缓存形状: {model.past_key_values[0][0].shape} {model.past_key_values[0][1].shape}')
        print(f'第二次预测: {tokenizer.batch_decode(pred[..., -1:])}')

    # 多步推理
    print("\n进行10步额外推理:")
    with torch.inference_mode():
        for i in range(10):
            output = model.base_model(pred[..., -1:], past_key_values=model.past_key_values, use_cache=True)
            pred = output.logits.argmax(-1)
            input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
            print(f'步骤 {i + 1}: {tokenizer.batch_decode(pred[..., -1:])}')

    print(f"\n生成结果 (最后12个token): {tokenizer.decode(input_ids[0][-12:])}")
    return input_ids


# 平衡状态演示
def equilibrium_demo(model, tokenizer, input_ids):
    print("\n===== 模型输出平衡状态演示 ======")

    with torch.inference_mode():
        output = model.base_model(torch.as_tensor(input_ids).cuda()).logits
        pred = output.argmax(-1)
        print(f'输入序列 (最后20个token): {tokenizer.batch_decode(input_ids[0][-20:])}')
        print(f'预测输出 (最后20个token): {tokenizer.batch_decode(pred[0, -20:])}')

    # 干扰平衡状态
    print("\n干扰平衡状态 - 将'village'替换为'mountain':")
    with torch.inference_mode():
        input_ids_mod = deepcopy(input_ids)
        input_ids_mod[0][-4] = 14378  # 替换'village'为'mountain'
        output = model.base_model(torch.as_tensor(input_ids_mod).cuda()).logits
        pred = output.argmax(-1)
        print(f'修改后输入 (最后20个token): {tokenizer.batch_decode(input_ids_mod[0][-20:])}')
        print(f'修改后输出 (最后20个token): {tokenizer.batch_decode(pred[0, -20:])}')


# Medusa头基础推理
def medusa_basic_demo(model, tokenizer, prompt, past_key_values):
    print("\n===== Medusa头基础推理 ======")

    # 显示Medusa头结构
    print("Medusa头结构:")
    print(model.medusa_head)

    with torch.inference_mode():
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).cuda()
        model.current_length_data.zero_()

        # 第一次推理
        medusa_logits, outputs, logits = model(input_ids, output_orig=True, past_key_values=past_key_values)
        print(f'Medusa logits形状: {medusa_logits.shape}, logits形状: {logits.shape}')

        # 获取预测结果
        medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim=-1)
        pred = torch.argmax(logits[..., -1, :], dim=-1)
        print(f'基础模型预测: {tokenizer.batch_decode(pred)}')
        print(f'Medusa预测: {tokenizer.batch_decode(medusa_pred)}')

        # 组合预测
        preds = torch.cat([pred, medusa_pred[:, 0]], dim=-1)
        print(f'组合预测: {tokenizer.batch_decode(preds)}')

        # 第二次推理
        print("\n第二次推理:")
        medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig=True,
                                               past_key_values=model.past_key_values)

        medusa_pred = torch.argmax(medusa_logits[..., -5:, :], dim=-1)
        pred = torch.argmax(logits[..., :, :], dim=-1)
        print(f'基础模型预测: {tokenizer.batch_decode(pred[0])}')
        print(f'截断输入token: {preds[1:].tolist()}')
        print(f'输出token: {pred[0, :].tolist()}')

        # 计算接受长度
        posterior_mask = (preds[1:] == pred[0, :-1]).int()
        accept_length = torch.cumprod(posterior_mask, dim=-1).sum().item()
        cur_length = accept_length + input_len + 1
        print(f'后验掩码: {posterior_mask.tolist()}')
        print(f'接受长度: {accept_length}')
        print(f'当前KV缓存长度: {model.current_length_data[0].item()}')
        print(f'开始长度: {input_len}, 当前长度: {cur_length}')

        # 更新KV缓存
        model.current_length_data.fill_(cur_length)

        # 创建新输入
        preds = torch.cat([pred[:, accept_length], medusa_pred[:, 0, accept_length]], dim=-1)
        print(f'组合预测: {tokenizer.batch_decode(preds)}')


# Medusa列表解码完整流程
def medusa_list_decoding(model, tokenizer, prompt, past_key_values):
    print("\n===== Medusa列表解码完整流程 ======")

    inference_count = 0
    accept_lengths = []

    with torch.inference_mode():
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).cuda()
        model.current_length_data.zero_()

        # 第一次推理
        medusa_logits, outputs, logits = model(input_ids, output_orig=True, past_key_values=past_key_values)
        inference_count += 1

        medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim=-1)
        pred = torch.argmax(logits[..., -1, :], dim=-1)
        preds = torch.cat([pred, medusa_pred[:, 0]], dim=-1)
        print(f'预测 @ {inference_count}: {tokenizer.batch_decode(pred)}')

        cur_length = input_len
        accept_lengths.append(1)

        # 多步推理
        max_steps = 50  # 限制步数，避免过长
        for i in range(max_steps):
            medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig=True,
                                                   past_key_values=model.past_key_values)
            inference_count += 1

            medusa_pred = torch.argmax(medusa_logits[..., -5:, :], dim=-1)
            pred = torch.argmax(logits[..., :, :], dim=-1)

            posterior_mask = (preds[1:] == pred[0, :-1]).int()
            accept_length = torch.cumprod(posterior_mask, dim=-1).sum().item()
            cur_length = cur_length + accept_length + 1

            # 更新KV缓存
            model.current_length_data.fill_(cur_length)

            # 创建新输入
            preds = torch.cat([pred[:, accept_length], medusa_pred[:, 0, accept_length]], dim=-1)
            print(f'预测 @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)

            # 检查结束条件
            if tokenizer.eos_token_id in pred[0, :accept_length + 1] or i == max_steps - 1:
                break

    # 绘制接受长度图
    plt.figure(figsize=(10, 5))
    plt.plot(accept_lengths)
    plt.xlabel('推理步骤')
    plt.ylabel('接受长度')
    plt.title('Medusa列表解码接受长度')
    print(f'平均接受长度: {np.mean(accept_lengths)}')

    return accept_lengths


# Medusa树注意力解码
def medusa_tree_decoding(model, tokenizer, prompt, past_key_values, past_key_values_data, current_length_data):
    print("\n===== Medusa树注意力解码 ======")

    accept_lengths_tree = []

    with torch.inference_mode():
        new_token = 0
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).cuda()
        model.current_length_data.zero_()

        reset_medusa_mode(model)
        medusa_choices = mc_sim_7b_63  # 使用预设的medusa choices

        # 生成Medusa缓冲区
        medusa_buffers = generate_medusa_buffers(
            medusa_choices, device=model.base_model.device
        )

        # 初始化Medusa
        medusa_logits, logits = initialize_medusa(
            input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        cur_length = input_len + 1
        accept_lengths_tree.append(1)

        # 多步树解码
        max_steps = 50  # 限制步数
        for i in range(max_steps):
            # 生成候选
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )

            # 树解码
            medusa_logits, logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # 评估后验
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature=0, posterior_threshold=0, posterior_alpha=0
            )

            # 更新推理输入
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            # 计算接受长度
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_lengths_tree.append(accept_length_tree)

            # 检查结束条件
            if tokenizer.eos_token_id in input_ids[0, input_len:] or i == max_steps - 1:
                break

    print(f'解码结果: {tokenizer.batch_decode(input_ids[:, input_len:])}')
    return accept_lengths_tree


# 比较列表解码和树解码
def compare_decoding_methods(accept_lengths_list, accept_lengths_tree):
    print("\n===== 解码方法比较 ======")

    # 计算移动平均以平滑曲线
    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # 绘制比较图
    plt.figure(figsize=(12, 6))

    # 原始数据（透明）
    plt.plot(accept_lengths_tree, label='树解码 (原始)', alpha=0.3)
    plt.plot(accept_lengths_list, label='列表解码 (原始)', alpha=0.3)

    # 平滑后的数据
    window_size = 5
    plt.plot(moving_average(accept_lengths_tree, window_size), label='树解码 (平滑)', color='tab:blue')
    plt.plot(moving_average(accept_lengths_list, window_size), label='列表解码 (平滑)', color='tab:orange')

    plt.xlabel('推理步骤')
    plt.ylabel('接受长度')
    plt.legend()
    plt.title('树解码 vs 列表解码性能比较')

    # 打印统计信息
    print(f'树解码平均接受长度: {np.mean(accept_lengths_tree)}')
    print(f'列表解码平均接受长度: {np.mean(accept_lengths_list)}')

    # 计算加速比


    accelerate_ratio = np.mean(accept_lengths_tree) / np.mean(accept_lengths_list)
    print(f'估计加速比: {accelerate_ratio:.2f}x')

    plt.show()


# 主函数
def main():
    # 配置
    gpu_id = "0"  # 使用的GPU ID
    # model_path = "/home/vision/work/codebase/paper-projects/axolotl-ctl/vicuna_7b_stage1"  # 模型路径
    model_path = "FasterDecoding/medusa-vicuna-7b-v1.3"
    medusa_num_heads = 4  # Medusa头数量
    use_local_base_model = True  # 是否使用本地基础模型路径

    # 设置环境
    setup_environment(gpu_id)

    # 加载模型和分词器
    model, tokenizer = load_model(model_path, medusa_num_heads, use_local_base_model)

    # 初始化KV缓存
    past_key_values, past_key_values_data, current_length_data = init_kv_cache(model)

    # 设置提示词
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"

    # 运行基本推理演示
    input_ids = basic_inference_demo(model, tokenizer, prompt)

    # 运行平衡状态演示
    equilibrium_demo(model, tokenizer, input_ids)

    # 重新初始化KV缓存
    past_key_values, past_key_values_data, current_length_data = init_kv_cache(model)

    # 运行Medusa头基础推理
    # medusa_basic_demo(model, tokenizer, prompt, past_key_values)

    # 重新初始化KV缓存
    past_key_values, past_key_values_data, current_length_data = init_kv_cache(model)

    # 运行Medusa列表解码
    accept_lengths_list = medusa_list_decoding(model, tokenizer, prompt, past_key_values)

    # 重新初始化KV缓存
    past_key_values, past_key_values_data, current_length_data = init_kv_cache(model)

    # 运行Medusa树解码
    accept_lengths_tree = medusa_tree_decoding(model, tokenizer, prompt, past_key_values, past_key_values_data,
                                               current_length_data)

    # 比较两种解码方法
    compare_decoding_methods(accept_lengths_list, accept_lengths_tree)


if __name__ == "__main__":
    main()
