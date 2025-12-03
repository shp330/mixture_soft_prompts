import os, pdb, sys
import random
from typing import Union, List, Dict

import torch
import torch.nn as nn
from torch import FloatTensor
from  torch.nn import Module
import torch.nn.functional as F
from assets.static_vars import ATTRIBUTE_TOKEN_LEN, MAX_MIXTURE_SIZE, device
num_to_string = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]

class SoftEmbedding(nn.Module):
  """ Performs traditional non-controlled data augmentation """

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    """appends learned embedding to original embedding
    Args:
      original_emb (nn.Embedding): original transformer word embedding
      n_tokens (int, optional): number of tokens for task. Defaults to 10.
      num_exemplars (int, optional): number of exemplars which determines the initialization text
      init_from_vocab (bool, optional): initalizes from default vocab.
      tokenizer: a tokenier for init_text
    """
    super().__init__()
    self.name = 'base-embedding'
    self.original_emb = original_emb
    self.n_tokens = n_tokens

    init_text = f"Show me {num_to_string[num_exemplars + 1]} distinct utterances that all express the "
    init_prompt_value = self.init_embedding(
      original_emb, n_tokens, init_from_vocab, tokenizer, init_text
    )
    self.soft_prompt = nn.Parameter(init_prompt_value, requires_grad=True).to(device)
    print(f"Initialized soft prompts with dimension {self.soft_prompt.shape}")

  def init_embedding(
      self,
      original_emb,
      n_tokens,
      init_from_vocab: bool,
      tokenizer,
      init_text
      ):
    """
    initializes learned embedding
      either from vocab, random initialization or a custom set of init_text

    Args:
        original_emb:
        n_tokens:
        init_from_vocab:
        tokenizer:
        init_text:

    Returns:
      torch.float: initialized using original schemes


    """

    if init_from_vocab:
      init_embd = self.original_emb.weight[:n_tokens].clone().detach()
      if tokenizer is not None:
        # replace the embedding with init_text from static vars
        tokens = tokenizer(init_text)
        for i, token in enumerate(tokens['input_ids']):
          init_embd[i] = self.original_emb.weight[token]
          if i + 1 >= init_embd.shape[0]:
            break
        print(f"Initialized embedding with '{init_text}'")
      else:
        print(f"Initialized embedding with tokens from the vocabulary")
    else:
      rr = 0.5 # random_range
      dimension = original_emb.weight.size(1)
      init_embd = torch.FloatTensor(n_tokens, dimension).uniform_(-rr, rr)
      print(f"Initialized embedding with random vectors")
    return init_embd

  def forward(self, tokens):
    raise NotImplementedError

  @classmethod
  def from_saved_embedding(cls, args, original_emb, prompt_path):
    if args.accelerate:
      weights = torch.nn.Parameter(torch.load(prompt_path).half())
    else:
      weights = torch.load(prompt_path)

    num_prompt_tokens = weights.shape[0]
    previous_embed = cls(original_emb, num_prompt_tokens)
    previous_embed.soft_prompt = weights
    print(f"Loaded prompt weights from {prompt_path}")
    return previous_embed

  def save_prompt_embedding(self, save_path, prompt_file):
    prompt_path = os.path.join(save_path, prompt_file)
    torch.save(self.soft_prompt, prompt_path)
    print(f"Saved a soft prompt at {prompt_path}")

class CausalEmbedding(SoftEmbedding):

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    super().__init__(original_emb, n_tokens, num_exemplars, init_from_vocab, tokenizer=tokenizer)
    self.name = 'causal-embedding'

  def forward(self, tokens):
    """run forward pass
    Args:
      tokens (torch.long): input tokens before encoding
    Returns:
      torch.float: encoding of text concatenated with learned task specifc embedding

    Reasoning: During the first pass, we are operating in the encoding phase, so we
      modify the input sequence to use the soft prompt.  In subsequent passes, we are
      now operating in the generation phase, so we just process the tokens normally.
      Since generation operates one token at a time, we check whether the sequence
      length is <= 1 token to recognize when we are in the generation phase.
    """
    batch_size, seq_len = tokens.shape
    # use soft prompt unless we are using the autoregressive `.generate()`
    if seq_len > 1:
      input_embed = self.original_emb(tokens[:, self.n_tokens:])
      learned_embed = self.soft_prompt.repeat(batch_size, 1, 1)
      final_embed = torch.cat([learned_embed, input_embed], 1)
    else:
      final_embed = self.original_emb(tokens)
    return final_embed

class Seq2SeqEmbedding(SoftEmbedding):

  def __init__(self, original_emb: nn.Embedding, n_tokens: int=10, num_exemplars: int=5,
              init_from_vocab: bool=True, tokenizer=None):
    super().__init__(original_emb, n_tokens, num_exemplars, init_from_vocab, tokenizer=tokenizer)
    self.name = 'seq2seq-embedding'

  def forward(self, tokens):
    """run forward pass
    Args:
      tokens (torch.long): input tokens before encoding
    Returns:
      torch.float: encoding of text concatenated with learned task specifc embedding

    Reasoning: During the first pass, we are operating in the encoding phase, which we
      recognize by checking that the first token in the first example contains a negative
      value.  This token_id == -1 since we manually set it as the placeholder earlier.
      When this is not the case, then we are in the generation phase, so we may simply
      proceed as normal with the original embedding.
    """
    if tokens[0][0] < 0:  # if first token is a soft prompt placeholder
      input_embed = self.original_emb(tokens[:, self.n_tokens:])
      learned_embed = self.soft_prompt.repeat(tokens.shape[0], 1, 1)
      final_embed = torch.cat([learned_embed, input_embed], 1)
    else:
      final_embed = self.original_emb(tokens)
    return final_embed


class AttributeAttention(nn.Module):
  """
  自定义属性注意力模块（基于线性投影 + 温度系数 + Einstein 求和的注意力机制）

    作用：对输入的嵌入向量（input_embed）做最大池化 + 线性投影 + 层归一化后，与混合特征（mixture）计算注意力分数
         再通过注意力权重对 mixture 做加权聚合，最终输出聚合后的混合特征。
    场景：常用于多属性 / 多模态【特征融合】、软提示（Soft Prompt）【权重聚合】等场景。
    Attributes:
          attn: 线性投影层。将输入嵌入映射到同维度空间（无偏置，避免引入额外偏移）
          attn_non_linear: 非线性激活：SiLU（Swish），比 ReLU更 平滑，保留梯度
          layer_norm: 层归一化。稳定训练，减少内部协变量偏移
          temperature: 温度系数：调节注意力分布的尖锐程度（值越小，注意力越集中）
          温度系数：
              温度↑: 注意力分布更均匀（权重分散）；
              温度↓: 注意力更集中（权重聚焦少数特征）
  """
  attn:Module
  attn_non_linear:Module
  layer_norm:Module
  temperature:float
  def __init__(self, in_dim, temperature):
    super().__init__()
    self.attn = nn.Linear(in_dim, in_dim, bias=False)
    self.attn_non_linear = nn.SiLU()
    self.layer_norm = nn.LayerNorm(in_dim) # 添加非线形，增强表达能力，应在 layer_norm 后使用。不添加激活则是线性投影，表达能力较弱
    self.temperature = temperature

  def forward(
      self,
      input_embed: FloatTensor,
      mixture: FloatTensor
      ) -> FloatTensor:
      """

      Args:
          input_embed: 输入嵌入向量，形状通常为 [seq_len, in_dim]（序列长度 × 特征维度）
          mixture: 混合特征/候选特征，形状通常为 [batch_size, n_props, in_dim]（批次 × 属性数 × 特征维度）
                   混合特征（如多属性候选特征、多组软提示权重）

      Returns:
          torch.float: 注意力加权聚合后的混合特征，形状为 [in_dim] 或 [batch_size, in_dim]

      """
      # First we project the input_embeding into a new space that fits with mixtures
      # We are learning to project the embedding such that multiplication with the mixture
      # produces attention scores
      # 步骤1：对输入嵌入做全局最大池化（沿序列维度，dim=0）
      #       max_pool_inputs_embeds 形状：[in_dim]（seq_len维度被池化，仅保留特征维度）
      #       _ 是最大池化对应的索引，此处无用
      # 最大池化作用：压缩输入嵌入的序列维度（如 [10, 768] → [768]），提取序列中最具代表性的特征；
      #             避免序列长度干扰注意力计算，聚焦嵌入的 “全局核心特征”。
      max_pool_inputs_embeds, _ = torch.max(input_embed, 0)

      # 步骤2：线性投影 + 层归一化（无激活？代码中注释了SiLU但未使用，需注意）
      x = self.attn(max_pool_inputs_embeds)  # 投影：[in_dim] → [in_dim]
      x = self.layer_norm(x)  # 层归一化：稳定特征分布
      # x = self.attn_non_linear(x)  # 代码中定义了 self.attn_non_linear 但未调用
      # now we get attention scores by mutipling mixture and the projection
      # softmax produces a weighting scheme

      # 步骤3：计算注意力分数（逐元素相乘 + 温度系数缩放）
      #       mixture 形状 [b, p, d]，x 形状 [d] → 广播为 [b, p, d] 后逐元素乘
      #       attn_scores 形状：[batch_size, n_props, in_dim]
      attn_scores = (mixture * x) / self.temperature

      # 步骤4：归一化注意力权重（dim=-1: 沿最后一维，最后一维和为 1）
      # normalized_attn_scores 形状：[batch_size, n_props, in_dim]，最后一维和为1
      normalized_attn_scores = F.softmax(attn_scores, -1)

      # 步骤5：Einstein求和实现注意力加权聚合
      # 'bpl, bpd -> pd' 含义：
      #    对 batch 维度求和，对 p（属性）和 d（维度）保留 → 输出 [n_props, in_dim]
      #    若 mixture 无 batch 维度，输出为 [in_dim]
      #    b: batch_size，
      #    p: n_props（属性数）
      #    l/d: in_dim（特征维度）
      mixture = torch.einsum('bpl, bpd -> pd', normalized_attn_scores, mixture)
      return mixture

class AttributeBottleneck(nn.Module):
  """
  带瓶颈层的属性注意力模块（基于「下投影→非线性→上投影」的瓶颈结构 + 注意力加权聚合），
  核心目标是：对输入嵌入（input_embed）做瓶颈层特征压缩 + 重构，生成注意力权重，
  再对混合特征（mixture）做跨批次的注意力加权聚合，最终输出聚合后的属性特征。
  Notes:
      瓶颈层（down+up 线性层）:通过特征维度压缩减少计算量、增强特征表达能力，是轻量化特征融合的典型设计。
  """
  def __init__(self, in_dim, hidden_dim, temperature):
    super().__init__()
    # 1. 瓶颈层下投影：将输入维度 in_dim 压缩到 hidden_dim（无偏置，避免偏移干扰）
    self.attn_W_down = nn.Linear(in_dim, hidden_dim, bias=False)
    # 2. 瓶颈层上投影：将压缩后的 hidden_dim 重构回 in_dim（恢复原维度）
    self.attn_W_up = nn.Linear(hidden_dim, in_dim, bias=False)
    # 3. 非线性激活：SiLU（Swish），比 ReLU 更平滑，保留梯度（瓶颈层非线性）
    self.attn_non_linear = nn.SiLU()
    # 4. 层归一化：稳定训练，减少内部协变量偏移（重构后归一化）
    self.layer_norm = nn.LayerNorm(in_dim)
    # 5. 温度系数：调节注意力分布的尖锐程度（值越小，注意力越集中）
    self.temperature = temperature

  def forward(self, input_embed, mixture):
      """

      Args:
          input_embed: 输入嵌入向量，形状 [seq_len, in_dim]（序列长度×特征维度）
          mixture: 混合特征/候选特征，形状 [batch_size, n_props, in_dim]（批次×属性数×特征维度）

      Returns:
          注意力加权聚合后的混合特征，形状 [n_props, in_dim]

      """
      # First we project the input_embeding into a new space that fits with mixtures
      # We are learning to project the embedding such that multiplication with the mixture
      # produces attention scores
      # ========== 步骤1：输入嵌入全局最大池化 ==========
      # 压缩序列维度（seq_len→1），提取全局核心特征
      # max_pool_inputs_embeds 形状：[in_dim]
      max_pool_inputs_embeds, _ = torch.max(input_embed, 0)
      # ========== 步骤2：瓶颈层特征压缩+重构 ==========
      # 下投影：in_dim → hidden_dim（特征压缩，减少计算）
      x = self.attn_W_down(max_pool_inputs_embeds)  # [hidden_dim]
      # 非线性激活：引入非线性表达能力（瓶颈层核心）
      x = self.attn_non_linear(x)                   # [hidden_dim]
      # 上投影：hidden_dim → in_dim（特征重构，恢复原维度）
      x = self.attn_W_up(x)                         # [in_dim]
      # 层归一化：稳定重构后的特征分布，避免梯度爆炸/消失
      x = self.layer_norm(x)                        # [in_dim]
      # now we get attention scores by mutipling mixture and the projection
      # ========== 步骤3：注意力分数计算与归一化 ==========
      # 逐元素相乘：mixture [b,p,d] × x [d] → 广播为 [b,p,d]，再除以温度系数
      attn_scores = (mixture * x) / self.temperature
      # softmax produces a weighting scheme
      # 归一化：沿最后一维（特征维度）做 softmax，权重和为 1
      normalized_attn_scores = F.softmax(attn_scores, -1)
      # ========== 步骤4：注意力加权聚合（跨批次） ==========
      # Einstein求和：对batch维度（b）求和，保留属性（p）和特征维度（d）
      mixture = torch.einsum('bpl, bpd -> pd', normalized_attn_scores, mixture)
      return mixture


class AttributeConvolution(nn.Module):
  """
  Mixes prompts through convolution
  基于二维卷积的提示（Prompt）混合模块，
  核心目标是：对多组软提示（Prompt）张量做卷积操作，将多组提示融合为单一提示张量。相比注意力机制的 “加权聚合”，
  卷积更擅长捕捉提示间的局部空间关联（把提示组视为 “特征层”，卷积核在提示维度 / Token 维度做滑动计算），
  适合结构化的提示混合场景。

  Notes:
      二维卷积的作用:
         nn.Conv2d 的 4 个维度定义为：[batch, channels, height, width]，对应到本模块：
             batch：固定为 1（提示组的批次维度）；
             channels：提示组数（卷积的 “通道” 维度，对应 max_stack_height）；
             height：每个提示的 Token 长度（ATTRIBUTE_TOKEN_LEN）；
             width：每个 Token 的嵌入维度（emb_dim）。
         卷积核（3×3）在 height × width（Token × 嵌入维度）上滑动，捕捉同一提示内不同 Token、同一 Token 不同维度的局部关联；
           而通道维度的降维（8→4→1）则融合不同提示组的信息，最终输出单通道（单组提示）。
      常数填充（torch.ones）的设计考量
          输入提示组数可能小于 max_stack_height（如 8 组最大，实际输入 5 组），未填充部分用 1 填充而非 0 的原因：
              0: 填充可能导致卷积梯度消失（ReLU 激活下 0 输入无梯度）；
              1: 填充是弱先验，既不主导特征，又能保证卷积层有有效输入。
  """
  def __init__(self, emb_dim, stack_height=MAX_MIXTURE_SIZE):
    super().__init__()
    self.emb_dim = emb_dim                 # 每个 Token 的嵌入维度（如 768）
    self.max_stack_height = stack_height   # 最大提示组数（卷积输入通道数）
    self.attr_len = ATTRIBUTE_TOKEN_LEN    # 每个提示的 Token 长度（如 10）
    self.cnn = nn.Sequential(
        # 卷积1：输入通道=最大提示组数 → 输出通道=最大组数//2，保持特征尺寸不变
        nn.Conv2d(
            self.max_stack_height,
            self.max_stack_height // 2,
            kernel_size=3,   # 3×3卷积核（在Token×嵌入维度滑动）
            padding=1        # 填充1 → 卷积后尺寸不变（(H+2P-K)/S +1 = H）
        ),
        nn.ReLU(),  # 非线性激活，增强表达能力
        # 卷积2：输入通道=最大组数//2 → 输出通道=1（融合为单组提示）
        nn.Conv2d(self.max_stack_height // 2, 1, kernel_size=3, padding=1),
        nn.ReLU()
    )

  def forward(self, x):
    """

    Args:
        x(FloatTensor): 输入提示张量，形状 [n_prompts, attr_len, emb_dim]
            - n_prompts：实际输入的提示组数（≤ max_stack_height）
            - attr_len：每个提示的 Token 长度（固定为ATTRIBUTE_TOKEN_LEN）
            - emb_dim：每个 Token 的嵌入维度
    Returns:
        (FloatTensor): 融合后的单一提示张量，形状 [attr_len, emb_dim]

    """
    # 步骤1：创建固定尺寸的填充张量（适配卷积输入格式）
    # 形状：[batch=1, in_channels=max_stack_height, attr_len, emb_dim]
    # 注：batch维度固定为1，因输入是单批次的提示组；值初始化为1（常数填充）
    padded_tensor = torch.ones((1, self.max_stack_height, ATTRIBUTE_TOKEN_LEN, self.emb_dim)).to(device)
    # 步骤2：将实际输入提示填充到固定张量中（未填充部分保留为1）
    # x.shape[0] = 实际提示组数
    padded_tensor[0, :x.shape[0], :, :] = x

    # 步骤3：卷积融合 → 挤压冗余维度
    # cnn输出形状：[1, 1, attr_len, emb_dim] → squeeze后变为 [attr_len, emb_dim]
    return self.cnn(padded_tensor).squeeze()


class AttributeEmbedding(nn.Module):
  """
  Attributes:
      multi_attribute (bool):  是否多属性
      attribute_map(Union[List[Dict[str, int]], Dict[str, int]]):
        - List[Dict[str, int]]: 有多个属性时
        - Dict[str, int]: 只有一个属性时
  """
  multi_attribute: bool
  attention: AttributeAttention
  bottleneck: AttributeBottleneck
  cnn_mixture: AttributeConvolution
  attribute_map: Union[List[Dict[str, int]], Dict[str, int]]
  def __init__(
      self, args, attributes: list, original_emb: nn.Embedding, num_sets: int=1,
          frozen: bool=False, tokenizer = None, attribute_init_texts = None):
    """ Introduces new custom parameters to represent the attributes
    Args:
      args (dict): group of argument flags
      attributes (list): decides how many unique embeddings to create
      original_emb (nn.Embedding): to be used when initializing from attribute name
      num_sets (int): number of attribute sets, if greater than 1 then we are
        encoding different types of attributes, such as intents and domains
      frozen (bool): if True, freeze the parameters to their initial encoding
      tokenizer: a tokenizer for init_text
      attribute_init_texts (list[str]): list of texts to initialize the attributes from
        must match the length of attributes
    """
    super().__init__()
    self.name = 'attribute-embedding'
    self.original_emb = original_emb
    self.multi_attribute = num_sets > 1

    self.instruction_prompt = None
    self.constraints = None
    self.attribute_map = None
    self.attribute_embedding = None

    self.mixture_type = args.mixture
    self.model_type = args.model

    if self.mixture_type == 'attention':
      self.attention = AttributeAttention(original_emb.weight.size(1), args.temperature)
      self.attention.to(device)

    if self.mixture_type == 'bottleneck':
      self.bottleneck = AttributeBottleneck(original_emb.weight.size(1), args.hidden_size, args.temperature)
      self.bottleneck.to(device)

    if self.mixture_type == 'cnn':
      stack_height = MAX_MIXTURE_SIZE
      if args.dataset == 'nlu++':
        stack_height = 6
      elif args.dataset == 'crossner':
        stack_height = 8
      elif args.dataset == 'topv2':
        stack_height = 10
      self.cnn_mixture = AttributeConvolution(
        original_emb.weight.size(1), stack_height=stack_height
      )
      self.cnn_mixture.to(device)

    if len(attributes) > 0:
      if self.multi_attribute:
        self.attribute_map = []
        self.attribute_embedding = []
        categories = ['intent', 'slot']  # Remove to generalize
        assert(len(categories) == len(attributes))

        for idx, attrs in enumerate(attributes):
          attr_map = {attr:j for j, attr in enumerate(attrs)}
          self.attribute_map.append(attr_map)
          category, attr_init_text = categories[idx], attribute_init_texts[idx]

          init_attr_values = self.initialize_tokens(len(attrs), tokenizer, attr_init_text)
          attr_embed = nn.Parameter(init_attr_values, requires_grad=not frozen).to(device)
          self.attribute_embedding.append(attr_embed)
          print(f"Initialized {category} tokens with dimension {attr_embed.shape}")

      else:
        self.num_attributes = len(attributes)
        self.attribute_map = {attr:idx for idx, attr in enumerate(attributes)}

        init_attr_values = self.initialize_tokens(len(attributes), tokenizer, attribute_init_texts)
        self.attribute_embedding = nn.Parameter(init_attr_values, requires_grad=not frozen).to(device)
        print(f"Initialized attribute tokens with dimension {self.attribute_embedding.shape}")

  def initialize_tokens(self, n_attributes, tokenizer=None, attribute_init_texts=None):
    """
    initializes learned embedding
    random_range (float, optional): range to init embedding, only applies
      when not initializing from vocab. Defaults to 0.5.

    Args:
        original_emb:
        n_attributes: 属性数量
        tokenizer: 分词器
        attribute_init_texts: 属性初始化文本

    Returns:
        (FloatTensor): initialized using original schemes

    """
    start, stop = 0, ATTRIBUTE_TOKEN_LEN
    init_embeds = []
    for _ in range(n_attributes):
      embed = self.original_emb.weight[start:stop].clone().detach()
      init_embeds.append(embed)

      start += ATTRIBUTE_TOKEN_LEN
      stop += ATTRIBUTE_TOKEN_LEN

    if attribute_init_texts:
      if not tokenizer:
        raise ValueError("tokenizer must be provided if attribute_init_texts is provided")
      if n_attributes != len(attribute_init_texts):
        raise ValueError(f"Number of attributes {n_attributes} does not match number of attribute_init_texts")

      for n in range (n_attributes):
        tokens = tokenizer(attribute_init_texts[n])
        for i, token in enumerate(tokens['input_ids']):
          init_embeds[n][i] = self.original_emb.weight[token]
          if i + 1 >= init_embeds[n].shape[0]:
            break

      print(f"Initialized embedding with texts for attribute embeds")
    return torch.stack(init_embeds)

  def forward(self, token_batch):
    """run forward pass
    Args:
      token_batch (torch.long): input tokens before encoding (batch_size x seq_len)
    Returns:
      final_embed (torch.float): encoding of text prepended with learned task specifc
      embeddings of shape (batch_size x seq_len x embed_dim)
    """
    if token_batch[0][0] < 0 or (self.model_type=='gpt' and token_batch.shape[1] > 1):
      final_embeddings = []
      instruct_embed = self.instruction_prompt.soft_prompt
      instruct_len = self.instruction_prompt.n_tokens

      for tokens, attributes, pad_len in zip(token_batch, self.constraints, self.pad_lengths):
        attr_len = self.calc_attribute_length(attributes)
        prefix_len = instruct_len + pad_len

        pad_embed = self.original_emb(tokens[instruct_len:prefix_len])
        input_embed = self.original_emb(tokens[prefix_len+attr_len:])
        attr_embed = self.embed_constraints(input_embed, attributes)
        final_embed = torch.cat([instruct_embed, pad_embed, attr_embed, input_embed])
        final_embeddings.append(final_embed)

      return torch.stack(final_embeddings).to(device)

    else: # do not use soft prompt if we are in the generation phase
      return self.original_emb(token_batch)

  @staticmethod
  def repeat_to_fill(descriptions, tokenizer):
    desc_embedding = tokenizer(descriptions)['input_ids']

    filled = []
    for tokens, description in zip(desc_embedding, descriptions):
      num_repeats = (ATTRIBUTE_TOKEN_LEN // len(tokens)) + 1  # add one so we go over
      filled.append( f"{description} " * num_repeats )
    return filled

  def _get_tokens(self, constraint_queries, level):
    """ given a list of attribute strings written in canonical form, will return a list of the
    attribute token embeddings for feeding into a model. """
    attribute_embeds = []
    for query in constraint_queries:
      if self.multi_attribute:
        attr_index = self.attribute_map[level][query]
        attr_embed = self.attribute_embedding[level][attr_index]
      else:
        attr_index = self.attribute_map[query]
        attr_embed = self.attribute_embedding[attr_index]
      attribute_embeds.append(attr_embed)
    return attribute_embeds

  def set_constraints(self, metadata):
    """
    set_constraints that will be used when running a forward pass
    If no constraints are set, only the instruction prompt is used with the domain
    sanity check that constraints are found within self.attribute_map keys.
    """
    for constraints in metadata['constraints']:

      if self.multi_attribute:
        num_sets = len(constraints)  # should be 2
        for index in range(num_sets):
          for category_constraint in constraints[index]:
            if category_constraint not in self.attribute_map[index].keys():
              raise ValueError(f'Constraint: {category_constraint} not in the mapping')

      else:
        # random.shuffle(constraints)  # if you want to shuffle the order
        for constraint in constraints:
          if constraint not in self.attribute_map.keys():
            raise ValueError(f'Constraint: {constraint} not in the ontology')

    self.constraints = metadata['constraints']
    self.pad_lengths = metadata['pad_lengths']

  def calc_attribute_length(self, constraint_set, level=-1):
    if len(constraint_set) == 0:
      return 0

    if self.multi_attribute and level < 0:
      con_sets, levels = constraint_set, [0,1]
      lengths = [self.calc_attribute_length(cs, lvl) for cs, lvl in zip(con_sets, levels)]
      attr_len = sum(lengths)
    else:
      if self.mixture_type == 'concat':
        attr_len = len(constraint_set) * ATTRIBUTE_TOKEN_LEN
      else:
        attr_len = ATTRIBUTE_TOKEN_LEN
    return attr_len

  def embed_constraints(self, input_embed, constraint_set, level=-1):
    if len(constraint_set) == 0:
      _, embed_dim = input_embed.shape
      active_device = input_embed.device
      return torch.empty((0, embed_dim), device=active_device)

    if self.multi_attribute and level < 0:
      con_sets, levels = constraint_set, [0,1]
      mixed = [self.embed_constraints(input_embed, cs, lvl) for cs, lvl in zip(con_sets, levels)]
      attr_embed = torch.concat(mixed)
    else:
      constraint_tokens = self._get_tokens(constraint_set, level)
      attr_embed = self.mix_operation(constraint_tokens, input_embed)
    return attr_embed

  def mix_operation(self, constraint_tokens, input_embed):
    attr_embed = torch.cat(constraint_tokens)  # (attr_token_len * n, embed_dim)

    if self.mixture_type == 'attention':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.attention(input_embed, joined_embed)
    elif self.mixture_type == 'bottleneck':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.bottleneck(input_embed, joined_embed)
    elif self.mixture_type == 'cnn':
      joined_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = self.cnn_mixture(joined_embed)
    elif self.mixture_type == 'pooling':
      stacked_embed = torch.stack(constraint_tokens, dim=0)
      attr_embed = torch.mean(stacked_embed, dim=0)

    return attr_embed

  @classmethod
  def from_saved_embedding(cls, args, original_emb, ckpt_path):
    if not ckpt_path:
      return cls(args, [], original_emb, frozen=True)

    prompt_file = ckpt_path.split('/')[-1]
    embedding_path = ckpt_path.replace(prompt_file, f"attr_vals_{prompt_file}")
    mapping_path = ckpt_path.replace(prompt_file, f"attr_map_{prompt_file}")

    num_sets = 2 if args.dataset == 'topv2' else 1
    previous_embed = cls(args, [], original_emb, num_sets, frozen=True)
    previous_embed.attribute_embedding = torch.load(embedding_path)
    previous_embed.attribute_map = torch.load(mapping_path)

    if args.mixture == 'attention':
      attention_path = ckpt_path.replace(prompt_file, f"attention_{prompt_file}")
      previous_embed.attention = torch.load(attention_path)
    elif args.mixture == 'bottleneck':
      bottleneck_path = ckpt_path.replace(prompt_file, f"bottleneck_{prompt_file}")
      previous_embed.bottleneck = torch.load(bottleneck_path)
    elif args.mixture == 'cnn':
      cnn_path = ckpt_path.replace(prompt_file, f"cnn_{prompt_file}")
      previous_embed.cnn_mixture = torch.load(cnn_path)

    print(f"Loaded prompt weights from {embedding_path} and {mapping_path}")
    return previous_embed

  def save_prompt_embedding(self, save_path, filename):
    attr_path = os.path.join(save_path, f"attr_vals_{filename}")
    torch.save(self.attribute_embedding, attr_path)

    attr_map = os.path.join(save_path, f"attr_map_{filename}")
    torch.save(self.attribute_map, attr_map)

    if self.mixture_type == 'attention':
      attention_path = os.path.join(save_path, f"attention_{filename}")
      torch.save(self.attention, attention_path)
    elif self.mixture_type == 'bottleneck':
      bottleneck_path = os.path.join(save_path, f"bottleneck_{filename}")
      torch.save(self.bottleneck, bottleneck_path)
    elif self.mixture_type == 'cnn':
      cnn_path = os.path.join(save_path, f"cnn_{filename}")
      torch.save(self.cnn_mixture, cnn_path)
    print(f"Saved attribute prompts at {attr_path} and {attr_map}")
