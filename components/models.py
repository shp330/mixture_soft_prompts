import os, pdb, sys
import numpy as np
import random
import json

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers import MaxLengthCriteria, StoppingCriteriaList, BeamSearchScorer, LogitsProcessorList
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import BertModel, GPT2LMHeadModel

from tqdm import tqdm as progress_bar
from collections import defaultdict
from assets.static_vars import device
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card_templates import ModelCardTemplate
from typing import List, Dict, Tuple, Type, Callable
    
class BaseModel(nn.Module):
  """
  基础模型抽象类，封装通用的分类头、dropout、激活函数等逻辑
  实现了基于 BERT/RoBERTa 的分类任务通用框架，封装了编码器、分类头、损失计算等核心逻辑

  Attributes:
      model_type: 模型类型
      weight_decay: 权重衰减系数

  Notes:
       包含编码器（self.encoder，如 RoBERTa/DeBERTa）、dropout 层（self.dropout）、全连接层（self.dense）、
         激活函数（self.gelu）、损失函数（self.criterion）、softmax（self.softmax）等基础组件；

  """
  def __init__(self, args, core, tokenizer):
    """

    Args:
        args: 配置参数对象，需包含以下字段：
            - model: 模型类型，支持 'bert'/'roberta'
            - verbose: 是否打印详细日志
            - debug: 调试模式开关
            - weight_decay: 权重衰减系数
            - drop_rate: dropout 概率
            - hidden_size: 分类头隐藏层维度
            - ont_size: 分类任务的类别数
        core: 预训练编码器核心（如 BertModel/RobertaModel）
        tokenizer: 对应的 tokenizer，用于辅助调试/日志输出
    """
    super().__init__()
    self.name = 'base-model'
    self.encoder = core
    self.model_type = args.model.lower()  # 统一小写，增强鲁棒性
    self.tokenizer = tokenizer

    # 配置参数
    self.verbose = args.verbose
    self.debug = args.debug
    self.weight_decay = args.weight_decay

    # 网络层
    self.dropout = nn.Dropout(args.drop_rate)

    self.dense = nn.Linear(core.config.hidden_size, args.hidden_size)
    self.gelu = nn.GELU()
    self.classify = nn.Linear(args.hidden_size, args.ont_size)

    # 损失函数与激活函数
    self.softmax = nn.LogSoftmax(dim=1) # 改为 dim=-1，适配任意维度
    self.criterion = nn.CrossEntropyLoss()  # performs LogSoftmax and NegLogLike Loss

  def forward(
      self,
      inputs: dict,
      targets,
      outcome='logit'
  ):
    """
    兼容RoBERTa/BERT 训练/推理模式（targets 可选）
    Args:
        inputs: 自 tokenizer 的编码结果，通常是字典，包含：
                - input_ids
                - attention_mask
                - token_type_ids（可能包含，但 RoBERTa 不用）
        targets: 真实标签，形状 [batch_size]，推理时可传 None
        outcome: 指定返回哪种形式的输出【当前代码只实现了 'logit' 和 'softmax'，没有处理 'log_softmax'】：
                - 'logit'：原始未归一化的分数（用于训练）
                - 'softmax'：概率分布（用于推理/预测）
                - 'log_softmax'：对数概率（常用于某些损失函数或 NLL）

    Returns:
        output: 模型输出（logit 或 概率）
        loss: 损失值（targets 为 None 时返回 None）

    """
    if self.model_type == 'roberta':
      """ By default, the encoder returns result of (batch_size, seq_len, vocab_size) under 'logits'
      When the output_hs flag is turned on, the output will also include a tuple under 'hidden_states'
      The tuple has two parts, the 【first is embedding output】 and the 【second is hidden_state of last layer】
      """
      enc_out = self.encoder(**inputs, output_hidden_states=True)
      cls_token = enc_out['hidden_states'][1][:, 0, :]
    # BERT 分支
    else:
      # 调用 BERT 编码器，获取最后一层所有 token 的表示。
      enc_out = self.encoder(**inputs)
      # 取第 0 个位置（即 [CLS] token）作为整个句子的表示。
      # cls_token 形状：[batch_size, hidden_size]
      cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim  BERT取最后一层CLS token

    # 典型的 MLP 分类头前向传播
    # 输入：[CLS] 向量（[batch_size, hidden_dim]）
    # 经过：Dropout → Linear → GELU → Dropout → Linear（分类层）
    # 输出：logits，形状应为 [batch_size, num_classes]
    hidden1 = self.dropout(cls_token)
    # 分类头采用 “线性层 + GELU+dropout” 的经典结构，兼顾表达能力和泛化性；
    hidden2 = self.dense(hidden1)
    hidden3 = self.gelu(hidden2)
    hidden4 = self.dropout(hidden3)
    logits = self.classify(hidden4)  # [batch_size, num_classes]
    # ！！可能导致维度丢失！！
    # 如果 num_classes == 1（如二分类用单输出），squeeze() 会把 [B, 1] 变成 [B]
    # 如果 num_classes > 1，squeeze() 无影响
    # 但 CrossEntropyLoss 要求 logits 是 [B, C]，target 是 [B]（long）
    # 所以 不应无条件 squeeze，建议删除这行。
    logits = logits.squeeze()

    # 当 targets is None（推理模式）时，这行会报错（因为 criterion 不能接受 None 作为 target）
    # loss = self.criterion(logits, targets) if targets is not None else None
    loss = self.criterion(logits, targets)
    # 持返回 logit（原始得分）或 softmax（概率），适配训练 / 推理不同需求
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss


def prepare_inputs_for_generation(
    input_ids,
    past=None,
    encoder_hidden_states=None,
    **kwargs
):
  """
  实现隐变量 z 注入生成过程
  Args:
      input_ids: 当前要输入模型的 token ID 序列（形状通常是 [batch_size, seq_len]）。
      past: 之前生成步骤中缓存的 key/value states（用于加速推理，避免重复计算），如果存在说明是增量解码（incremental decoding）。
      encoder_hidden_states: 用于 seq2seq 模型。在编码器-解码器架构（如 BART、T5）中，编码器输出的隐藏状态，供解码器 cross-attention 使用。
      **kwargs: 其他可选参数，如 attention_mask、token_type_ids、position_ids、use_cache 等。

  Notes:
       用于在自回归文本生成（如使用 Transformer 解码器或编码器-解码器架构）过程中，为模型的每一步生成准备好输入数据。
         它特别适用于 Hugging Face Transformers 库中某些模型（比如 GPT、BART、T5 等）
         在调用 .generate() 方法时的内部逻辑。
       支持不同架构：
         - 自编码器（如BERT）不需要此优化
         - 自回归模型（如GPT）需要此优化
         - 编码器-解码器（如T5/BART）需要encoder_hidden_states
       position_ids：位置 id 在 Transformer 模型中用于表示每个 token 在序列中的位置信息。
         Transformer 没有“顺序”概念，其自注意力机制是置换不变的（permutation-invariant），
         为了解决这个问题，原始 Transformer 论文引入了 位置编码（Positional Encoding），告诉模型每个 token 的位置。
         在 Hugging Face 的大多数模型（如 GPT-2、BERT、RoBERTa、BART 等）中，
           position_ids 就是用来索引位置嵌入（position embeddings）的整数张量，形状与 input_ids 相同。
      两种位置编码实习方式：
        - 绝对位置编码（如 BERT、GPT）：使用 position_ids 作为索引，从一个可学习或固定的 embedding 表中查出对应位置向量，加到 token embedding 上。
        - 相对位置编码（如 T5、DeBERTa）：不直接使用 position_ids，而是在注意力计算中显式建模 token 之间的相对距离。

  Returns:

  """
  token_type_ids = kwargs.get("token_type_ids", None)
  # 1. 处理past存在时的输入裁剪
  # 作用：当有过去的key-value缓存时，只需输入最后一个token
  # 原因：之前所有token的计算结果已缓存在past中，无需重复计算
  # only last token for inputs_ids if past is defined in kwargs
  # 如果 past 存在（即不是第一步生成），说明只需要最后一个 token 作为当前输入（因为前面的 token 已经通过 past 缓存了）。
  # ✅这是高效生成的关键：避免重复输入整个历史序列。
  if past:
    input_ids = input_ids[:, -1].unsqueeze(-1)
    # 同样地，如果提供了 token_type_ids，也只保留最后一个位置的值。
    if token_type_ids is not None:
      token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

  attention_mask = kwargs.get("attention_mask", None)
  position_ids = kwargs.get("position_ids", None)
  
  # 2. 动态创建位置ID
  # 如果用户没有提供 position_ids，但提供了 attention_mask，就根据 attention mask 自动生成 position IDs。
  if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past:
      position_ids = position_ids[:, -1].unsqueeze(-1)
  else:
    position_ids = None
  # !!!!!!!!!!!!!!!!!!! start: modified vs original, to pass inputs_embeds when they are available
  model_inputs = {"input_ids": input_ids}
  if encoder_hidden_states is not None:
    model_inputs["encoder_hidden_states"] = encoder_hidden_states
  model_inputs.update({
    "past_key_values": past, # 用于 KV 缓存加速
    "use_cache": kwargs.get("use_cache"), # 是否启用缓存（通常生成时为 True）
    "position_ids": position_ids, # 位置编码索引
    "attention_mask": attention_mask, # 注意力掩码（用于区分真实 token 和 padding）
    "token_type_ids": token_type_ids, # 用于区分句子 A/B（如 BERT 风格）
  })
  return model_inputs

class CVAEModel(BaseModel):
  """
  基于条件变分自编码器（CVAE）的可控文本生成模型，融合了 BERT 作为编码器、GPT-2 作为解码器，
    并引入了 隐变量采样（VAE）机制 和 束搜索（Beam Search）生成策略

    - Beam Search：通过 num_beams 和 beam_scorer 实现束搜索，提升生成质量。
    - CVAE（条件变分自编码器）：在标准 VAE 基础上，加入“条件信息”（如输入文本），使得生成过程可以受控。
  用途：可控文本生成（如风格迁移、对话生成、摘要等）
  CVAE 模型的目标：
    - 从输入文本中提取语义表示（通过 BERT/RoBERTa 编码器）
    - 学习隐变量 z：用 VAE 的方式学习一个潜在变量 z，即，隐变量（代表句子的压缩语义）。
        通过 VAE 从编码器输出采样，引入可控随机性。通过重参数化技巧从均值和方差中采样，引入随机性，提升多样性。
    - z 可用于后续任务（如文本生成、风格迁移、对话生成等）

  Notes:
      架构：
        - 编码器（BERT）：将输入文本编码为上下文感知的表示。
        - 通过 mu_linear / logvar_linear 构建 VAE 的隐空间
        - 解码器（GPT-2）：改造版 GPT-2，支持 cross-attention（看编码器）+ 接收 z。
            基于隐变量 z 和可能的条件信息（如 BERT 编码结果），自回归地生成目标文本。
        - 自定义 prepare_inputs_for_generation 实现隐变量 z 注入生成过程
      生成控制（支持 Beam Search + 温度采样 + 重复惩罚）：
        - 使用 StoppingCriteriaList + BeamSearchScorer 控制生成长度与搜索策略
        - 利用 logits_warpers（temperature/top-p）和 logits_processors（重复惩罚等）提升文本质量
  """
  def __init__(self, args, encoder, decoder_config, decoder, tokenizer):
    """

    Args:
        args: 包含超参数（如温度、beam 数、最大长度等）。
        encoder: BERT 编码器。
        decoder_config:
        decoder: 改造后的 GPT-2，支持 cross-attention（即能关注编码器输出）。
        tokenizer: 用于文本与 token ID 互转。
    """
    super().__init__(args, encoder, tokenizer)
    self.name = 'cvae-model'
    self.embedder = encoder.embeddings  # BERT嵌入层，后续可能用于构造初始输入。
    self.config = decoder_config
    # 被修改过的 GPT-2 ，支持 cross-attention，即能接收来自编码器（BERT）的 key/value，实现 encoder-decoder 架构（类似 BART/T5）。
    self.decoder = decoder  # GPT2解码器（改造后支持cross attention）
    # 关键定制点：覆盖 GPT-2 默认的 prepare_inputs_for_generation 方法。
    #           替换解码器的生成功能的输入处理逻辑（适配自定义嵌入+隐变量z）
    # 原因：标准 GPT-2 是纯 decoder，不处理隐变量 z 或 encoder hidden states。
    # 自定义函数需在生成时注入：
    #   隐变量 z（拼接到输入嵌入或作为额外条件）：通过 VAE 从编码器输出采样，引入可控随机性
    #   encoder 的输出（用于 cross-attention）
    #   past_key_values 等缓存机制
    self.decoder.prepare_inputs_for_generation = prepare_inputs_for_generation

    # VAE隐变量建模：均值/方差层（实现隐变量z的采样）
    # 从 BERT 编码器的 [CLS] 表示（或其他聚合表示）预测隐变量 z 的分布参数
    #   - mu = self.mu_linear(h)
    #   - logvar = self.logvar_linear(h)
    # 然后通过【重参数化技巧（reparameterization trick）】 采样：
    self.mu_linear = torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)
    self.logvar_linear = torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)

    # 生成相关配置（beam search、停止条件、logits处理）
    # 1、Stopping Criteria（停止条件）
    #    通过 MaxLengthCriteria 限制生成文本长度，生成达到 target_max_len 时自动停止。
    self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.target_max_len)])

    # 2、Beam Search（束搜索）：使用 BeamSearchScorer 实现束搜索。
    #   注意：beam_scorer_batch_size 应等于输入 batch size × num_beams（此处命名可能有误，
    #          通常 batch_size 是原始输入 batch，不是 num_generations）。
    #   实际使用时需确保输入张量已扩展为 batch_size * num_beams。
    self.num_beams = args.num_generations
    self.beam_scorer_batch_size = args.num_generations
    self.beam_scorer = BeamSearchScorer(self.beam_scorer_batch_size, num_beams=self.num_beams, device=device)

    # 3. Logits Warpers（采样策略）扭曲器（控制生成策略：温度等）
    #    控制生成的随机性：
    #      - temperature：调节 softmax 的平滑度（<1 更确定，>1 更随机）
    #      - 此处 top_k=0, top_p=1.0 表示不使用 top-k/top-p 采样，仅用温度缩放。
    self.logits_warpers = self.decoder._get_logits_warper(temperature=args.temperature, top_k=0, top_p=1.0)

    # 4. Logits Processors 后处理约束（控制生成策略：重复惩罚）
    #    - 重复惩罚：repetition_penalty（值 >1 抑制重复 token）
    #    - n-gram 重复禁止：no_repeat_ngram_size（如设为 2，则禁止连续 2-gram 重复）
    self.logits_processors = self.decoder._get_logits_processor(
            repetition_penalty=args.threshold,
            no_repeat_ngram_size=decoder.config.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=None,
            input_ids_seq_length=None,
            encoder_input_ids=None,
            bad_words_ids=None,
            min_length=None,
            max_length=None,
            eos_token_id=None,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=None,
            num_beam_groups=None,
            diversity_penalty=None,
            remove_invalid_values=None,
            exponential_decay_length_penalty=None,
            logits_processor=LogitsProcessorList(),
            renormalize_logits=None, 
            suppress_tokens=None, 
            begin_suppress_tokens=None, 
            forced_decoder_ids=None, 
        )

  def get_input_embeddings(
      self,
      input_ids,
      attention_mask
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 BERT/RoBERTa 编码器中提取（⚠️注意：RoBERTa/BERT 的 [CLS] token 位于序列开头（index 0），通常用于句子级表示）：
      - 词嵌入矩阵 H0（即输入 token 的初始嵌入，未经 Transformer 编码）
      - [CLS] token 的最终隐藏状态 h_0_last（经过整个编码器后的表示）

    适配 RoBERTa/BERT 编码器（无token type ids）

    Args:
        input_ids: 输入token的id序列，shape=(B, seq_len)
        attention_mask: 注意力掩码，shape=(B, seq_len)

    Returns:
        - H0: 输入词嵌入矩阵，shape=(B, seq_len, hidden_size)
        - h_0_last: CLS token的最终隐藏状态，shape=(B, hidden_size)

    """
    # 1. 提取词嵌入（避免重复计算，复用编码器内置的嵌入层）
    # detach().clone() 防止梯度回传至原始input_ids，
    #   input_ids 是 LongTensor，本身不可导，.detach().clone() 可能略显冗余，但无害。
    # roberta does not use token type ids
    # H0: 初始词嵌入（未经过 Transformer），用于后续可能的重建或解码
    H0 = self.embedder(input_ids=input_ids.detach().clone()) # (batch, seq_len, hidden_size)
    # encode, get h_0_last
    H0 = H0.to(device)
    # last_hidden_state：最后一层所有 token 的表示，形状 (B, seq_len, hidden_size)
    # [:, 0]：取第 0 个 token（即 [CLS]）作为句子表示，输出形状：(B, hidden_size)
    h_0_last = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0] # (Batch, hidden_size)
    # h_0_last = self.encoder(input_embeds=H0, attention_mask=attention_mask).last_hidden_state[:,0] # (Batch, hidden_size) # we can just pass in input ids and token type ids, itll do input embeds internally
    return H0, h_0_last

  def reparameterize(self, h_0_last):
    """
    VAE重参数化技巧：从采样隐变量z
    实现 VAE 的重参数化技巧，从后验分布 q(z∣x)=N(μ,σ^2)【即，N(mu, var)】 中采样隐变量 z。
    Args:
        h_0_last: 句子表示，即，最后一层隐藏层的第一个 token 表示，即 [cls]

    Returns:
        - z: 采样得到的隐变量，形状 (B, 1, hidden_size)
        - logvar, mu: 对数方差和均值，用于计算 KL 散度损失（VAE 损失的一部分）

    """
    # reparameterize to get z
    # z = mu + exp(0.5 * logvar) * ε,   ε ~ N(0, I)
    # z 的维度与 BERT 隐藏层相同（例如 768），可直接用于初始化解码器状态或拼接输入。
    # 计算均值 mu 和 对数方差 logvar；mu, logvar 形状均为 (B, hidden_size)
    mu = self.mu_linear(h_0_last)
    logvar = self.logvar_linear(h_0_last)
    std = (0.5 * logvar).exp()
    # 从标准正态分布采样噪声 eps
    eps = torch.randn_like(mu)
    z = mu + eps * std  # (B, hidden_size)
    # 添加一个序列维度（长度为 1），便于与词嵌入拼接或作为解码器初始状态
    #   常见于：将 z 作为额外 token 输入解码器，或初始化 RNN/LSTM 隐藏状态
    z = z[:, None, :]  # (B, 1, hidden_size)
    return z, logvar, mu

  def decode(self, input_ids, attention_mask, labels, z):
    """
    作用：从编码的潜在表示 z 和输入文本中提取信息，生成目标文本（labels）的表示，通过解码器生成最终输出。
    该函数是CVAE（条件变分自编码器）+ GPT2 解码器 架构中的核心解码逻辑，核心目标是：
      - 从完整输入序列中分离「条件部分」和「生成目标部分」；
      - 将 CVAE 采样的隐变量 z 融合到条件的词嵌入中；
      - 调用 GPT2 解码器完成生成任务（训练时计算损失，推理时生成文本）；

    将隐变量 z 与生成目标部分（response 或 reply）的词嵌入融合后，
      作为 GPT-2 解码器的 cross-attention 输入，从而引导生成过程。
    输入格式为：[条件文本] [SEP/EOS] [目标文本]，例如对话中的 [历史] [SEP] [回复]。
    隐变量 z 是从 CVAE 的编码器（如 BERT + VAE 头）采样得到的，
      shape 为 (B, 1, hidden_size)，代表整个条件-目标对的潜在语义。

    融合隐变量z与解码器输入，调用GPT2解码器计算损失/输出    BERT编码 + VAE均值/方差计算

    Args:
        input_ids:输入文本的 token IDs，shape=(B, seq_len)
          输入序列结构：[条件部分] + [SEP/EOS token] + [生成目标部分]（RoBERTa 中 SEP 和 EOS token ID 相同）
        attention_mask: 输入文本的注意力掩码，shape=(B, seq_len)
        labels: 解码器的目标标签（用于计算损失），shape=(B, seq_len)
        z: CVAE采样的隐变量（潜在变量）/编码表示，shape=(B, 1, hidden_size)

    Returns:
        解码器输出（包含logits、loss等）

    """
    # decode
    # For RoBERTa, sep token id is eos token id
    # ========== 1. 分离条件部分和生成目标部分（基于SEP/EOS token） ==========
    # 找到输入中分隔符（如 [SEP]或 <eos>）的位置，作为「条件部分」和「生成目标部分」的分割点；
    #   （RoBERTa中 SEP token ID = EOS token ID，找到每个样本的第一个 SEP/EOS 位置）
    # +1 是为了获取分隔符后的第一个 token 位置
    # 结果：(B,) 形状的张量，表示每个批次中分隔符后的索引
    indices_of_sep = 1 + (input_ids == self.tokenizer.eos_token_id).max(dim=1).indices # (B, seq_len)

    #  ========== 2. 构建条件部分的注意力掩码（y_attn_mask） ==========
    # 创建全零掩码
    y_attn_mask = torch.zeros(attention_mask.shape)
    # 在每个样本的分隔符后的位置标记为 1
    y_attn_mask[(torch.arange(attention_mask.shape[0]), indices_of_sep)] = 1  # SEP后一位标记为1
    # 累积和：SEP 及之前位置全为0，SEP之后位置全为1
    y_attn_mask = y_attn_mask.cumsum(dim=1).to(device)
    # 反转：分隔符后的部分为 0，前面为 1
    y_attn_mask = 1 - y_attn_mask

    # ========== 3. 提取条件信息 ==========
    # 只保留分隔符之前的 token（条件信息），分隔符之后的部分用 pad_token 填充
    # 结果：y_ids 只包含条件部分
    y_ids = torch.where(
        y_attn_mask == 1,
        input_ids,
        self.tokenizer.pad_token_id
    ).to(device)

    # ========== 4. 嵌入条件信息 ==========
    # 将条件 token IDs 转换为嵌入向量，形状：(B, cond_seq_len, hidden_size)
    H0_y = self.embedder(input_ids=y_ids)

    # get H12'
    # 将潜在变量 z与条件嵌入连接：H0_y[:,1:]可能跳过了第一个 token（如 [CLS]）
    decoder_inputs = torch.cat((z, H0_y[:,1:]), dim=1) # (B, y_seq_len, hidden_size)

    # 增强z的融合：将 z在序列维度重复，并与原输入拼接。目的：让每个时间步都能访问完整的潜在表示 z
    #   - 原始：(B, seq_len, hidden_size)
    #   - 重复 z：(B, seq_len, hidden_size)
    #   - 拼接后：(B, seq_len, hidden_size * 2)
    _, seq_len, _ = decoder_inputs.shape
    # 这是一种更强的融合方式：每个位置的表示都显式包含全局隐变量 z。
    decoder_inputs = torch.cat((decoder_inputs, z.repeat(1, seq_len, 1)), dim=-1) # (B, y_seq_len, hidden_size*2)

    # ========== 6. 处理解码器标签（忽略PAD部分的损失） ==========
    # -100 是 HuggingFace 的约定，表示忽略该位置的 loss。
    d_labels = torch.where(labels==self.tokenizer.pad_token_id, -100, labels)

    # 构建标签的注意力掩码（PAD位置为0）
    # labels_attn_mask 用于 decoder 的 self-attention（防止看到 future tokens？但 GPT-2 自带 causal mask）。
    labels_attn_mask = torch.where(labels == self.tokenizer.pad_token_id, 0, 1)

    # ========== 7. 调用GPT2解码器前向传播 ==========
    outputs = self.decoder(
        input_ids=labels,  # 目标序列
        attention_mask=labels_attn_mask,  # 标签的注意力掩码
        encoder_hidden_states=decoder_inputs,  # 条件信息 + 潜在变量
        labels=d_labels, # 计算损失用的标签
        encoder_attention_mask=y_attn_mask  # 只关注条件部分
    )

    return outputs

  def forward(self, input_ids, attention_mask, labels):
    #  1. 编码器：获取CLS token的隐藏态（h_0_last）
    _, h_0_last = self.get_input_embeddings(input_ids, attention_mask)

    # 2. 重参数化：采样隐变量z（核心VAE逻辑）
    # reparameterize：通过均值（mu）和方差（logvar）采样隐变量 z，保证梯度可回传；
    z, logvar, mu = self.reparameterize(h_0_last)
    # 3. 解码器：基于z和输入生成文本，计算生成损失
    # decode：将 z 拼接至解码器输入，结合 GPT2 的生成逻辑计算损失；
    outputs = self.decode(input_ids, attention_mask, labels, z)

    # https://arxiv.org/pdf/1312.6114.pdf
    # 4. 损失融合：生成损失 + KL散度（VAE的核心损失）
    # 损失由两部分组成：生成损失（GPT2 的 LM 损失） + KL 散度（约束 z 服从标准正态分布）。
    kld = -0.5 * (
        1 + logvar - mu ** 2 - logvar.exp()
    ).sum()  # or avg before sum? https://github.com/shj1987/ControlVAE-ICML2020/blob/master/Language_modeling/Text_gen_PTB/model.py
    outputs.loss += kld

    return outputs

  def generate(self, input_ids, attention_mask, **kwargs):
    """
    生成方法：beam search 生成
    生成特点：
      生成阶段从标准正态分布采样 z，实现 “可控 + 多样” 的文本生成；
      采用 beam search 提升生成质量，支持自定义停止条件、温度系数等
    Args:
        input_ids:
        attention_mask:
        **kwargs:

    Returns:

    """
    input_ids = input_ids.to(device)
    H0 = self.embedder(input_ids=input_ids) #, token_type_ids=token_type_ids) # (B, seq_len, hidden_size)\
    B, seq_len, hidden_size = H0.shape # should be just the z and y
    # 1. 随机采样隐变量z（生成阶段无监督信号，直接采样）
    z = torch.randn((B, hidden_size)).to(device) #  (B, hidden_size)
    z = z[:, None, :] # (B, 1, hidden_size)
    # 2. 构造解码器输入：拼接z和编码器嵌入
    decoder_inputs = torch.cat((z, H0[:,1:]), dim=1) # (B, seq_len, hidden_size)
    decoder_inputs = torch.cat((decoder_inputs, z.repeat(1, seq_len, 1)), dim=-1).to(device) # (B, seq_len, hidden_size*2)
    decoder_inputs = decoder_inputs.repeat_interleave(self.num_beams, dim=0)
    encoder_attention_mask = attention_mask.repeat_interleave(self.num_beams, dim=0)
    # https://github.com/huggingface/transformers/issues/6535#issuecomment-1353658984
    starter_inputs = torch.tensor([[self.tokenizer.cls_token_id]]*B*self.num_beams).to(device)

    if B != self.beam_scorer_batch_size: # Could also drop last batch
        beam_scorer = BeamSearchScorer(batch_size=B, num_beams=self.num_beams, device=device)
    else:
        beam_scorer = self.beam_scorer
    # 3. Beam search生成
    outputs = self.decoder.beam_sample(starter_inputs, beam_scorer=beam_scorer, encoder_hidden_states=decoder_inputs, \
            encoder_attention_mask=encoder_attention_mask, stopping_criteria=self.stopping_criteria, \
            pad_token_id=self.config.pad_token_id, logits_processor=self.logits_processors, logits_warper=self.logits_warpers, **kwargs)
    return outputs

  def resize_token_embeddings(self, new_size):
    self.decoder.resize_token_embeddings(new_size)
    self.encoder.resize_token_embeddings(new_size)

  def to(self, device):
    self.encoder.to(device)
    self.decoder.to(device)
    self.mu_linear.to(device)
    self.logvar_linear.to(device)

  def save_pretrained(self, path):
    torch.save({'model_state_dict': self.state_dict(),
        'ckpt_name': self.config.name_or_path,
        'encoder_model_name': self.encoder.name_or_path,
        'decoder_model_name': self.decoder.name_or_path}, path)
    
  @staticmethod
  def from_pretrained(args, path, tokenizer):  
    # 如果 path 不包含 'bert'（例如是本地保存的 .pt 文件）：
    if 'bert' not in path:
      prev_weights = torch.load(path)
      ckpt_name = prev_weights['ckpt_name']
      encoder_model_name = prev_weights['encoder_model_name']
      decoder_model_name = prev_weights['decoder_model_name']
    else:  
      encoder_model_name = path
      decoder_model_name = 'gpt2'
    encoder_config = AutoConfig.from_pretrained(encoder_model_name)
    encoder_config.vocab_size = len(tokenizer)
    encoder = BertModel(encoder_config)
    decoder_config = AutoConfig.from_pretrained(decoder_model_name)
    decoder_config.hidden_size *= 2 # 将隐藏层维度翻倍（可能是为了匹配编码器输出 + 潜在变量拼接后的维度）。
    decoder_config.is_decoder = True # 启用 decoder 模式（允许 attention mask 等）。
    decoder_config.add_cross_attention = True # 添加 cross-attention 层，使解码器能关注编码器的输出（这是 Seq2Seq 架构的关键）
    decoder_config.vocab_size = len(tokenizer)
    decoder = GPT2LMHeadModel(decoder_config) # 实例化带语言模型头的 GPT-2：GPT2LMHeadModel。 这里只是随机初始化的 GPT-2，权重后续由 load_state_dict 加载。

    model = CVAEModel(args, encoder, decoder_config, decoder, tokenizer)
    if 'bert' not in path:
      # 加载预训练权重（如果存在）
      # 仅当从 checkpoint 加载时（即 path 不是原始 BERT 名称），才加载保存的模型状态字典。
      model.load_state_dict(prev_weights['model_state_dict'])
    model.to(device)
    return model

class DualClassifier(BaseModel):
  """
  双标签分类模型（DualClassifier），同时预测 intent（意图）和 slot（槽位），损失求和训练；
    目标是共享编码器和隐藏层参数，同时完成两个分类任务：
    - intent（意图分类）：粗粒度的对话意图判断（如 “查询天气”“订机票”）；
    - slot（槽位分类）：细粒度的实体 / 属性提取（如 “时间”“地点”“金额”）；
  通过共享底层参数降低模型参数量，同时将两个任务的损失求和进行联合训练，适配对话系统的核心需求。

  Notes:
      模型选型：根据args.size自动选择编码器类型（小 / 中型用 RoBERTa，其他用 DeBERTa）；
      核心设计：“共享编码器 + 双分类头”，是多任务学习中 “参数共享” 的经典范式，兼顾效率和任务关联性。
  """
  # Model for predicting both intents and slots at once
  def __init__(
      self,
      args,
      core,
      tokenizer,
      primary_size,
      secondary_size
  ):
    """
    继承父类BaseModel，复用编码器、激活函数、损失函数等基础组件；
    动态选择编码器类型（适配不同规模的预训练模型）；
    定义两个独立的线性分类头，分别对应 intent 和 slot 任务，共享底层参数但任务输出层分离。

    Args:
        args: 配置参数（如 hidden_size 隐藏层维度、size 模型规模）
        core: 核心模型组件（通常是预训练编码器）
        tokenizer: 分词器（RoBERTa/DeBERTa tokenizer）
        primary_size: 主分类的类别数（intent 任务分类头数）
        secondary_size: 次分类的类别数（slot 任务分类头数）
    """
    super().__init__(args, core, tokenizer)
    self.name = 'dual-classify'

    # 根据参数选择编码器类型（RoBERTa/DeBERTa）
    self.model_type = 'roberta' if args.size in ['small', 'medium'] else 'deberta'
    # 定义双分类头：线性层实现分类（输入为隐藏层维度，输出为类别数）
    self.primary_classify = nn.Linear(args.hidden_size, primary_size)  # intent分类头
    self.secondary_classify = nn.Linear(args.hidden_size, secondary_size)  # slot分类头

  def forward(self, inputs, targets, outcome='logit'):
    """
    共享编码器和隐藏层，降低参数量；
    双分类头分别预测意图（intent）和槽位（slot），损失求和训练；
    输出为字典格式，区分两个任务的预测结果，适配对话系统的核心需求
    Args:
        inputs: 编码器输入（字典，包含 input_ids、attention_mask 等）
        targets: 标签字典，包含'intent'和'slot'两个键，对应标签值
        outcome: 输出类型（'logit'返回原始预测值，其他返回概率）

    Returns:
        output: 字典，包含intent/slot的预测结果（logit/概率）
        loss: 联合损失（intent_loss + slot_loss）

    """
    # 共享编码器，双分类头分别计算损失
    enc_out = self.encoder(**inputs, output_hidden_states=True)
    # 取最后一层的第 0 个 token（CLS token）—— 这是 BERT/RoBERTa 类模型中用于句子级分类的核心特征，能表征整个输入序列的语义；
    # 若任务是 “槽位标注”（序列级分类，如每个 token 对应一个槽位），通常会提取所有 token 的隐藏状态，
    #   但此处代码提取 CLS token，说明该 slot 任务是句子级槽位分类（如整句的核心槽位），而非逐 token 标注。
    cls_token = enc_out['hidden_states'][-1][:, 0, :]
    # 共享隐藏层计算
    # 两个分类任务共享这部分特征变换，大幅降低参数量（若分开设计，需两套隐藏层），
    #   同时让 intent 和 slot 任务的特征相互关联（符合对话任务中 “意图决定槽位，槽位支撑意图” 的逻辑）。
    hidden1 = self.dropout(cls_token)
    hidden2 = self.dense(hidden1)
    hidden3 = self.gelu(hidden2)
    hidden4 = self.dropout(hidden3)

    # 双分类头与损失联合训练
    # 两个线性分类头独立，但输入是同一特征hidden4，实现 “共享底层、分离顶层”；
    intent_logits = self.primary_classify(hidden4)
    slot_logits = self.secondary_classify(hidden4)         # batch_size, num_classes

    # 损失求和：loss = intent_loss + slot_loss，训练时梯度会同时更新编码器、共享隐藏层、两个分类头的参数，
    #   让模型同时优化两个任务；
    # 若两个任务的损失量级差异大，可加权重（如loss = 0.6*intent_loss + 0.4*slot_loss），代码中未体现，是简化版。
    intent_loss = self.criterion(intent_logits, targets['intent'])
    slot_loss = self.criterion(slot_logits, targets['slot'])
    loss = intent_loss + slot_loss

    # 输出原始logit（训练时用，便于计算梯度）
    # 训练阶段：outcome='logit'，返回原始 logit（未经过 softmax），
    #   因为损失函数（如 CrossEntropyLoss）会自动对 logit 计算 softmax，避免重复计算；
    if outcome == 'logit':
      output = {'intent': intent_logits, 'slot': slot_logits}

    # 推理阶段：outcome≠'logit'，返回 softmax 归一化后的概率，便于直接取最大概率作为预测类别。
    else:
      output = {'intent': self.softmax(intent_logits), 'slot': self.softmax(slot_logits)}
    return output, loss

class GenerateModel(BaseModel):
  """
  通用分类模型，直接用 CLS token 作为 logits，适用于快速验证

  Notes:
      特点：
        - 直接使用 CLS token 的输出作为 logits，无需额外分类层
        - 适用于快速验证模型架构和数据有效性
        - 支持分类/回归任务（通过配置 task_type）
  """
  # Main model for general classification prediction
  def __init__(self, args, core, tokenizer):
    super().__init__(args, core, tokenizer)
    self.name = 'generate'

  def forward(self, inputs, targets, outcome='logit'):
    """

    Args:
        inputs: 模型输入字典，包含 input_ids, attention_mask 等
        targets: 标签数据，shape [batch_size]
        outcome: 输出类型，'logit'返回原始得分，'prob'返回概率

    Returns:
        output: logits 或概率值，shape [batch_size, hidden_dim]
        loss: 损失值（如果 targets 不为 None）

    """
    enc_out = self.encoder(**inputs)
    cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    logits = cls_token
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss

class SentenceBERT(SentenceTransformer):
  """
  扩展训练 / 相似度评估逻辑
  扩展了 qualify 方法，用于可视化句子间的余弦相似度，方便调试和验证句子嵌入（embedding）的效果

  Notes:
      SentenceBERT 继承自 SentenceTransformer（Sentence-BERT 官方库的核心类，用于生成句子嵌入），
        因此拥有父类的所有能力（如加载预训练模型、生成嵌入等）。
      这个 qualify 方法是 Sentence-BERT 模型的调试工具，通过随机抽样 + 相似度排序 + 可视化输出，
        快速验证句子嵌入的语义表征能力，帮助开发者判断模型是否学到了合理的语义相似性关系。
  """
  def qualify(self, features, utterances):
    """
    可视化句子相似度，便于调试和效果验证；
    作用：通过随机选一个句子，计算它与所有其他句子的余弦相似度，
           最终打印 “最相似的 3 个句子” 和 “最不相似的 3 个句子”，直观验证嵌入效果。

    Args:
        features: 包含句子嵌入的字典，核心键是'sentence_embedding'，对应tensor类型的嵌入矩阵
        utterances: 与嵌入对应的原始句子列表（字符串列表）

    Returns:
        None（仅打印相似度结果，无返回值）

    """

    # 随机选一个句子，计算与其他句子的余弦相似度，输出最相似/最不相似的句子
    chosen_id = random.randint(0, len(utterances) - 1)
    chosen_utt = utterances[chosen_id]
    # 取出目标句子的嵌入向量，并通过 unsqueeze(0) 增加一个维度（从 [d] 变为 [1, d]，
    #   适配余弦相似度计算的维度要求）
    chosen_embed = features['sentence_embedding'][chosen_id].unsqueeze(0)

    comparables = []
    for sent_embed, utterance in zip(features['sentence_embedding'], utterances):
      with torch.no_grad():
        # 计算目标句子嵌入与当前句子嵌入的余弦相似度（取值范围 [-1, 1]，值越大越相似）。
        score = torch.cosine_similarity(chosen_embed, sent_embed.unsqueeze(0))
      comp = (utterance, round(score.item(), 3))
      comparables.append(comp)
    # 按相似度分数（元组的第二个元素）降序排序：分数最高的句子排在最前，最低的在最后。
    comparables.sort(key=lambda x: x[1], reverse=True)

    # 打印结果（可视化）
    print("Target utterance:", chosen_utt)
    print(f"Out of {len(utterances)} utterances, the 3 closest are:")
    count = 1
    # 最相似的 3 个句子：从 1 开始，因为最相似的是其本身，忽略
    for close, score in comparables[1:4]:
      print(f"   {count})", close, score)
      count += 1
    print(f"And the three furthest are:")
    count = 1
    # 最不相似的 3 个句子
    for far, score in comparables[-3:]:
      print(f"   {count})", far, score)
      count += 1

  def fit(self, train_objective: Tuple[object, nn.Module],
      evaluator, epochs: int = 1,
      steps_per_epoch = None,
      scheduler_name: str = 'WarmupLinear',
      warmup_steps: int = 10000,
      optimizer_class = optim.AdamW,
      optimizer_params : Dict[str, object]= {'lr': 3e-5},
      weight_decay: float = 0.01,
      logging_steps: int = 0,
      evaluation_steps: int = 0,
      output_path: str = None,
      save_best_model: bool = True,
      max_grad_norm: float = 3,
      do_qual: bool=False,
      callback: Callable[[float, int, int], None] = None,
      checkpoint_path: str = None,
      checkpoint_save_steps: int = 2000,
      checkpoint_save_total_limit: int = 0,
      args=None):
    """
    # 扩展训练逻辑：支持自定义损失函数、梯度裁剪、 checkpoint保存、训练中评估
    # 核心：
    # 1. 优化器分组（区分weight decay）：对 bias/LayerNorm 不做 weight decay，提升训练稳定性。
    # 2. 训练中随机选批次做相似度评估（do_qual）；
    # 3. 自定义学习率调度器、梯度归一化；
    # 4. 保存最佳模型（基于evaluator得分）。
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Only accepts on tuple now.
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model 
            performance during training on held-out dev data. Used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), 
            one epoch is equal the DataLoader size from train_objectives.
    :param scheduler_name: Learning rate scheduler. Available schedulers: 
            constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), 
            the learning rate is increased from o up to the maximal learning rate. 
            After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param logging_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param evaluation_steps: If > 0 and do qualify print out the closest relations per batch
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: 梯度裁剪的最大范数。Used for gradient normalization.
    :param do_qual: 是否在训练中随机选批次做相似度评估
    :param callback: 评估后回调函数，参数：(score, epoch, steps)。Callback function that is invoked after each evaluation.
        It must accept the following three parameters in this order:
        `score`, `epoch`, `steps`
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Checkpoint保存步数间隔。Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: 保留的Checkpoint最大数量（0表示不限制）。Total number of checkpoints to store
    """

    ##Add info to model card
    dataloader, loss_model = train_objective
    # # 生成模型卡片信息
    info_loss_functions =  ModelCardTemplate.get_train_objective_info(dataloader, loss_model)
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])
    eval_name = evaluator.__class__.__module__
    
    info_fit_parameters = {"evaluator": eval_name, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "scheduler": scheduler_name, "warmup_steps": warmup_steps, "weight_decay": weight_decay,
        "optimizer_class": str(optimizer_class), "optimizer_params": optimizer_params, 
        "evaluation_steps": evaluation_steps, "logging_steps": logging_steps, "max_grad_norm": max_grad_norm}
    print(info_fit_parameters)
    ifp = json.dumps(info_fit_parameters, indent=4, sort_keys=True)

    # 更新模型卡片
    self._model_card_text = None
    self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", ifp)
    self.best_score = -9999999

    self.to(self._target_device)
    loss_model.to(self._target_device)

    # Use smart batching
    # 使用智能批处理
    dataloader.collate_fn = self.smart_batching_collate
    # 计算训练步数
    if steps_per_epoch is None or steps_per_epoch == 0:
      steps_per_epoch = len(dataloader)
    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 优化器分组（区分weight decay）
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler = self._get_scheduler(optimizer, scheduler=scheduler_name, 
              warmup_steps=warmup_steps, t_total=num_train_steps)

    # 训练状态初始化
    global_step = 0
    data_iterators = []
    tok = self._first_module().tokenizer
    # 修复：随机批次选择应基于dataloader长度
    max_batch_idx = len(dataloader) - 1 if len(dataloader) > 0 else 0
    for epoch in progress_bar(range(epochs), desc="Epoch", total=epochs):
      training_steps = 0
      loss_model.zero_grad()
      loss_model.train()
      chosen_batch = random.randint(0, max_batch_idx) # len(dataloader)

      losses = []
      for features, labels in dataloader:
        # 终止条件：达到每轮步数限制
        if training_steps>=steps_per_epoch:
            break
        # 标签类型转换（适配损失函数）
        if labels.dtype == torch.int64:
          labels = labels.type(torch.float32)

        loss_value = loss_model(features, labels)
        losses.append(loss_value.item())

        if args.loss_function == 'default':
          torch.set_grad_enabled(False)
        else:
          loss_value.backward()

          # 梯度裁剪
          torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()

        # 步数更新
        training_steps += 1
        global_step += 1

        # 日志打印
        if logging_steps > 0 and training_steps % logging_steps == 0:
          avg_loss = round(np.mean(losses), 3) 
          print(f"Step {training_steps}/{steps_per_epoch}, Loss: {avg_loss}")

        # 保存Checkpoint
        if checkpoint_path is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
          print("Saving checkpoint")
          self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        # 相似度评估（随机批次）
        if do_qual and training_steps == chosen_batch:
          fzero = features[0]
          utterances = tok.batch_decode(fzero['input_ids'], skip_special_tokens=True)
          self.qualify(fzero, utterances)

      # 本轮训练结束：计算平均损失
      avg_loss = round(np.mean(losses), 3) if losses else 0.0
      def caller(raw_score, epoch, steps):
        score = round(raw_score, 3)
        print(f"Step {steps}/{steps_per_epoch}, Loss: {avg_loss}, Score: {score}")
      self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, caller)

    # 训练结束：保存最终Checkpoint
    if checkpoint_path is not None:
      self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

class SingleClassifier(AutoModelForSequenceClassification):
  """
  单标签分类模型
  适配 Hugging Face 标准分类接口，开箱即用。
  """
  def __init__(self, *args, **kwargs):
    super(SingleClassifier, self).__init__(*args, **kwargs)
    self.name = 'single-classify'
