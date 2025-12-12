import json
import random
import numpy as np

prompt_one = """Show me {size} distinct utterances that all express the following attributes of {attribute} in the {domain} domain. {explanations}"""
prompt_two = """Give me examples of {size} utterances that all include the {attribute} attributes in the {domain} domain. {explanations}"""
prompt_three = """You are a helpful assistant in the {domain} domain.  You will generate {size} example utterances that each include the attributes of {attribute}. {explanations}"""
prompt_four = """We are going to perform data augmentation today. Given two example utterance, you will generate {size} new examples that all include the required attributes of {attributes} within the {domain} domain. {explanations}"""


def explain_description(args, attributes, meanings, num_attributes, values):
  """
  生成针对数据集属性（attributes）的自然语言解释文本，并根据不同数据集（topv2/crossner）的特性调整最终返回的解释内容
  Args:
      args: 包含数据集名称等配置的对象（如args.dataset指定当前处理的数据集）；
      attributes: 属性名称列表（如["age", "gender"]）；
      meanings: 属性对应含义的列表（需与attributes长度一致，如["年龄信息", "性别信息"]）；
      num_attributes: 属性数量（注：实际代码中未直接使用该参数，可能是预留 / 冗余参数）；
      values: 关键词列表（仅crossner数据集会用到）。

  Returns:

  """
  if args.dataset == "topv2":
    return ""
  # parts用于存储单个属性的解释片段
  parts = []
  # 为每个属性生成格式为"属性名 refers to 属性含义"的解释字符串，并添加到parts中；
  for attr, meaning in zip(attributes, meanings):
    part = attr + " refers to " + meaning
    parts.append(part)
  explanation = join_together(parts, num_attributes) + ". "
  if args.dataset == "crossner":
    explanation += f"The resulting utterance must include the following keywords {', '.join(values)}."
  
  return explanation

def make_prompts(args, example, engineer, ontology):
  """
  面向不同 NLP 数据集（nlu++/crossner/topv2）构建提示词：
    - 从数据集本体（ontology）中提取属性 / 值的含义: 生成属性列表、含义列表及属性数量；
    - 拼接属性字符串和属性解释字符串；
    - 结合模板生成基础指令，并补充示例（exemplars），最终返回完整的 Prompt。

  Args:
      args: 全局配置对象（包含数据集名称、领域、生成数量等，如args.dataset/args.domain）
      example: 单条样本数据（字典格式，含attributes/values/domain等字段）
      engineer: 示例工程化相关参数（推测用于示例筛选，函数内仅传给find_exemplars_by_label）
      ontology: 数据集本体字典（存储属性 / 值的含义映射，分general（通用）和特定领域）

  Returns:

  """
  attributes = []
  meanings = []
  num_attributes = 0
  # 遍历样本的attributes字段，根据数据集类型从 ontology 中匹配属性的含义：
  for attr_name in example['attributes']:
    # 合并本体中「通用 intents」和「当前领域 intents」，取属性对应的含义；
    if args.dataset == "nlu++":
        attr_dict = ontology["general"]["intents"]
        attr_dict_domain = ontology[args.domain]["intents"]
        attr_dict.update(attr_dict_domain)  # 合并通用+领域专属intent含义
        meaning = attr_dict[attr_name]
        meanings.append(meaning)
    # 优先取当前领域的属性含义，不存在则用通用含义；
    elif args.dataset == "crossner":
        try:
          meaning = ontology[args.domain][attr_name] # 优先取领域专属含义
        except KeyError:
          meaning = ontology['general'][attr_name] # 兜底取通用含义
        meanings.append(meaning)
    attributes.append(attr_name)
    num_attributes += 1

  for value_name in example["values"]:
    if args.dataset == "nlu++":
      value_dict = ontology["general"]["slots"]
      value_dict_domain = ontology[args.domain]["slots"]
      value_dict.update(value_dict_domain)
      meaning = value_dict[value_name]
      meanings.append(meaning)
      attributes.append(value_name)
      num_attributes += 1
    elif args.dataset == "crossner":
      continue
    # 其他数据集：仅收集属性名，不处理含义（如后续的topv2）；
    elif args.dataset == "topv2":
      attributes.append(value_name)
      num_attributes += 1

  attribute_string = join_together(attributes, num_attributes)
  explain_string = explain_description(args, attributes, meanings, num_attributes, example['values'])

  instruction = prompt_one.format(size=args.num_generations, attribute=attribute_string, 
                                  domain=example['domain'], explanations=explain_string)
  exemplars = find_exemplars_by_label(args, example, engineer)
  with_exemplars = join_numbers(exemplars)
  return instruction + with_exemplars

def join_together(parts, size):
  if size == 1:
    return parts[0]
  elif size == 2:
    return parts[0] + " and " + parts[1]
  elif size > 2:
    suffix = f", and {parts[-1]}"
    prefix = ', '.join(parts[:-1])
    return prefix + suffix


def find_exemplars_by_label(
    args,
    sample: dict,
    engineer
):
  """
  Finds exemplars for a given sample and its constraints
  用于筛选少样本 Prompt 中的示例：
    从指定的示例池中（engineer.domain_data）筛选与目标样本（sample）属性匹配的候选示例，
      最终返回符合筛选规则的示例列表，用于补充到 Prompt 中提升生成效果。
  Args:
      args: 全局配置对象（如数据集、示例数量等）；
      sample: 目标样本（字典格式，含attributes/uuid等字段）；
      engineer: 示例工程化管理对象（封装了示例池、属性筛选规则、匹配逻辑等）；

  Returns:

  """
  exemplar_pool = []
  # 筛选目标样本的属性列表，仅保留 engineer.desired_attrs 中的属性，过滤无关属性；
  sample_attributes = [attr for attr in sample['attributes'] if attr in engineer.desired_attrs]

  for candidate in engineer.domain_data:
    # 跳过与目标样本uuid相同的候选（避免选到自身作为示例）；
    if candidate['uuid'] == sample['uuid']: continue

    # filter out generic attributes and values
    # 对每个候选示例，同样筛选核心属性（逻辑与目标样本一致）
    cand_attributes = [attr for attr in candidate['attributes'] if attr in engineer.desired_attrs]
    # decide if the candidate matches the sample
    cand_match, cms = engineer._overlap(sample_attributes, cand_attributes)
    # if both constraints are met, then add the candidate to the pool
    if cand_match:
      exemplar_pool.append((candidate, cms))
  # 对 exemplar_pool 中的候选示例做最终筛选，按匹配度排序、取前 N 个
  return sample_from_candidates(args, exemplar_pool, engineer)

def sample_from_candidates(args, exemplar_pool, engineer):
  """
  对初步匹配的示例池（exemplar_pool）进行「洗牌→按匹配度排序→截取候选池→随机抽样」，最终返回指定数量的示例，
    既保证示例质量（按匹配度排序），又引入随机性避免过拟合。
  Args:
      args: 全局配置对象（含pool_size/num_shot/model等关键配置）
      exemplar_pool: 初步匹配的示例池（元组列表：[(示例1, 匹配度1), (示例2, 匹配度2), ...]）
      engineer: 示例工程化对象

  Returns:

  """
  if len(exemplar_pool) == 0:
    raise ValueError("exemplar_pool is empty!")

  pool_size = args.pool_size
  # num_shot：需要的示例数量
  # pool_size：默认候选池大小
  # 当「需要的示例数量」大于「默认候选池大小」且「模型非 API 型」时，将 pool_size 扩容至 num_shot；
  # 设计目的：确保候选池有足够多的高质量示例（匹配度高）供抽样，避免因候选池过小导致抽样数量不足。
  if args.num_shot > pool_size and args.model != 'api':
    pool_size = args.num_shot

  random.shuffle(exemplar_pool)
  exemplar_pool.sort(reverse=True, key=lambda x: x[1])
  top_exemplars = [exemplar for exemplar, score in exemplar_pool][:pool_size]
  size = min(len(top_exemplars), args.num_shot)
  # 无放回随机抽样（replace=False）指定数量的示例；
  # 无放回：确保每个示例仅被选中一次，避免重复；
  # 随机性：保证每次抽样结果不同，提升 Prompt 的泛化性；
  chosen = np.random.choice(top_exemplars, size=size, replace=False)
  return list(chosen)

def join_numbers(exemplars):
  """
  少样本示例的格式化拼接工具函数
  作用：将筛选后的示例列表（exemplars）转为「带数字编号的文本字符串」，并在最后追加一个空的编号项（用于引导模型生成新文本），
       是 Prompt 构建中示例展示的标准化格式。
  Args:
      exemplars:

  Examples:
        >>> exemplars=[{"text": "I want to book a hotel"}, {"text": "Show me nearby restaurants"}]
        1) I want to book a hotel
        2) Show me nearby restaurants
        3)

  Returns:

  """
  exemplar_text = ""
  for i, exemplar in enumerate(exemplars):
    # 拼接后格式示例： 1) I want to book a hotel
    exemplar_text += f"\n{i + 1}) {exemplar['text']}"
  # 在示例列表末尾追加空的下个编号项
  # 目的：引导模型在该空编号后生成符合格式的新文本（Prompt 的关键引导技巧）；
  exemplar_text += f"\n{len(exemplars) + 1}) "
  return exemplar_text