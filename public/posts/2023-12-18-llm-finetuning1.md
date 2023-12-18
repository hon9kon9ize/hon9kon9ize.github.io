---
title: 調教大語言模型三步曲之一：揀預訓練模型
image: /images/llm-finetuning1.jpg
description: 今次文章係三步曲中嘅第一步，重點會講點揀預訓練模型嚟微調任務。
updated: 2023-12-18 10:35GMT
author: Joseph Cheng
---

![調教大語言模型三步曲之一：揀預訓練模型](/images/llm-finetuning1.jpg)

今次文章係三步曲中嘅第一步，重點會講點揀預訓練模型嚟微調任務。調教大語言模型前，先搞清楚啲術語先：「預訓練（Pre-training）」、「微調（Fine-tuning）」。文章標題係調教大語言模型，調教意指微調而非預訓練，預訓練係一個非常重要而且唔係一般人可以做得到嘅嘢，比如 Meta 嘅 [Llama-2](https://github.com/microsoft/Llama-2-Onnx/blob/main/MODEL-CARD-META-LLAMA-2.md) 7B 模型用咗二萬億個 tokens（唔多解釋咩係 token，你當係一個 token 等如一個中文字先），訓練咗 184,320 GPU 小時，呢個規模只有大公司先可以花費得起。微調相對嚟講用嘅資源就可以好平民化，有一張家用 GPU，最好有 24GB vram 就可以玩得起。

Meta 嘅 Llama-2 出咗好多版本，好似 [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) 係我所講嘅預訓練模型，即是未經過微調，本身冇對話或處理任務嘅能力，只係識估下一隻 token 係咩；而 [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) 故名思意係被微調到識傾偈嘅模型，一般如果模型名有 chat 都表示做咗微調。除咗傾下偈當然仲有好多唔同嘅任務可以訓練佢，有啲人會專微調佢成為翻譯、記摘要、或者 roleplay 噉呢啲專門任務，而好似 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 呢類微調語料就係包括多種廣泛任務。

點解要調教自己嘅模型呢？ChatGPT 都夠晒勁啦。其實好多場景都需要自己去 run 一個大語言模型，我所見其中一個最多人噉做嘅原因係 censorship 問題，當你問 ChatGPT 點樣暪住老婆去出軌，佢會苦口婆心叫你唔好噉做，大家都知有好多話題都會牽涉道德問題，作為一個面向大眾嘅服務 OpenAI 一定要把關，但如果我想做一個婚姻輔導嘅 Chatbot，需要問及好深入婚姻及兩性問題時，佢就叫你「咪啦」噉，點搞…，另外私隱問題都係好多人關注嘅，呢啲種種原因令好多開源微調模型應運而生。

## 點揀預訓練模型？

要調教出嚟效果好有幾個因素要考慮：
1. 目標語言係咪流利（廣東話）
2. 模型有冇具備所需知識（香港文化背景同歷史等）
3. 模型有冇良好理解同推理能力

我哋可以用以下嘅測試去評估邊個模型最啱使。

### 困惑度（Perplexity）

預訓練模型係由一大柞字訓練出嚟，模型從中學到常識、邏輯同語言能力，要知道模型對某啲語言或文字內容係咪熟識可以用 perplexity 嚟評估，因為預訓練模型訓練過程係逐隻逐隻預測出最大可能嗰柞字，形成一個機率分佈，如果比一柞字佢逐隻字去估，用估出嚟嘅 loss 就知佢係咪「識」甚至可能佢訓練嗰陣見過呢柞字。而家好多時都用嚟個方法去 check 下個模型係咪用咗 benchmark testset 去訓練嚟作弊。

```python
# 我揀咗幾個模型，冇揀 Mistral / Phi-2 呢啲勁 models 因為佢哋講到明冇用中文做預訓練。
models = [
	"01-ai/Yi-6B",
	"01-ai/Yi-34B,
	"THUDM/chatglm3-6b-base",
	"meta-llama/Llama-2-7b-hf",
	"Qwen/Qwen-7B",
	"indiejoseph/cantonese-llama-2-7b", # 呢個係我哋用粵維基 second pretrain Meta''s Llama-2
]

model_name = models[0]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
	 model_name,
	 device_map='auto',
	 load_in_4bit=True, # 我張 3090 4bit 先行到 34B 模型
	 torch_dtype=torch.bfloat16, # 要 check 下你張卡係咪對應
	 trust_remote_code=True # 某啲 model 需要
).eval()
```

```python
import torch
from tqdm import tqdm
from datasets import load_dataset

# 我用成個粵維基畀佢求出 perplexity
test = load_dataset("indiejoseph/wikitext-zh-yue", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# 之後叫個 model 行晒咁多行字
max_length = model.config.max_length
stride = 512
seq_len = encodings.input_ids.size(1)
device = "cuda"

nlls = []
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
	end_loc = min(begin_loc + max_length, seq_len)
	trg_len = end_loc - prev_end_loc # may be different from stride on last loop
	input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
	target_ids = input_ids.clone()
	target_ids[:, :-trg_len] = -100

	with torch.no_grad():
		outputs = model(input_ids, labels=target_ids)
		# loss is calculated using CrossEntropyLoss which averages over valid labels
		# N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
		# to the left by 1.
		neg_log_likelihood = outputs.loss

	nlls.append(neg_log_likelihood)

	prev_end_loc = end_loc
	if end_loc == seq_len:
		break  

# 得出 perplexity
ppl = torch.exp(torch.stack(nlls).mean())
```

#### 結果：

| model  | perplexity |
| ------- | --------- |
| 01-ai/Yi-6B | 81.5  |
| 01-ai/Yi-34B | 55.2 |
| THUDM/chatglm3-6b-base | inf |
| Qwen/Qwen-7B | 880640 |
| meta-llama/Llama-2-7b-hf | 37.2 |
| indiejoseph/cantonese-llama-2-7b | 22.9 |

數字越低代表個模型越有把握估到晒啲字。cantonese-llama-2 係用咗粵維基做二次預訓練，所以必定係最低 perplexity，可以用嚟作為一個參考指標；之後係 meta-llama/Llama-2-7b-hf 非常出色，Llama 2 paper 中有提過語料中有 0.13% 係 zh 而英文係佔 89.70% 共 27種語言；我本來對 Qwen 好大期望，但結果反映佢好可能用咗純簡中作訓練，所以 perplexity 低；ChatGLM 直頭 Inf，即時錯到離譜？或者 code 有 bug，唔深究喇；之後 Yi 6B／34B 都表現唔錯。

### 句子接龍

Perplexity 低只反映到個機率分佈靚，但唔係一定會文筆流暢，所以最好要佢寫啲嘢嚟睇下先。

```python
def text_generate(prompt, temperature=0.9):
	input_ids = tokenizer.encode(prompt, return_tensors="pt")
	output = model.generate(input_ids, max_length=100, num_return_sequences=1, repetition_penalty=1.2, do_sample=True, temperature=temperature, top_k=50, top_p=0.95)  

	return tokenizer.decode(output[0], skip_special_tokens=True)

# 畀般字佢開個頭之後要佢生成嚟緊成段字
prompt = "大家好，我係你哋嘅特首李家超，我熱愛"

print(text_generate(prompt))
```

#### 結果：

**01-ai/Yi-6B**
> 大家好，我係你哋嘅特首李家超，我熱愛香港、心繫香港。作為首長，我要為這座福地締造更多溫暖和諧的共融城市；作為大家共同的公務員團隊總領袖，我要帶領全體同事發揮最大效力，以服務市民為依歸，竭盡所能提高生活質素。

**01-ai/Yi-34B**
> 大家好，我係你哋嘅特首李家超，我熱愛香港這個地方同埋每一個市民。在慶祝回歸祖國25週年呢一日，我們全港人民一同見證新時代、新的起點。

**THUDM/chatglm3-6b-base**
> 大家好，我係你哋嘅特首李家超，我熱愛體育運動。在政府推動的各種體育活動中，我想為香港市民帶來更多好的體驗和更佳的健康狀況！  

**Qwen/Qwen-7B**
> 大家好，我係你哋嘅特首李家超，我熱愛中國文化並深信中華文明在這個世代裏有獨一無二的地位與影響力。香港是一個多元文化的都市，我很欣賞各種不同的文化和信仰。  

**meta-llama/Llama-2-7b-hf**
> 大家好，我係你哋嘅特首李家超，我熱愛中文讀者、愛書法及收集美術作品。 nobody is perfect, so let's love everyone!

**indiejoseph/cantonese-llama-2-7b**
> 大家好，我係你哋嘅特首李家超，我熱愛香港有史以來最富人民自由、樸守公義同法治社會嗰陣。1987年初，葛量外任行政局主席時期，就正式提出攞一部《基本法》，

以上結果係由每個 model 幾次生成結果中揀出嚟，可以見到 Llama 2 perplexity 係好低，但生成出嚟成日都夾雜英文，而且唔太識用廣東話詞匯；cantonese-llama-2 好啲，但內容唔通順；Yi ／ Qwen 非常流暢而且上文下理通順，但係變咗書面語。

### 問答比賽

雖然預訓練模型未經訓練去做對答任務，但只要個 prompt 寫成一段問答內容，佢都會順應噉生成答案。

```python
prompt = """香港是一個香港特別行政區是中華人民共和國的一個享有高度自治權的地方行政區域，直轄於中央人民政府。

誰是香港特別行政區的行政長官？

A: 李家超
B: 林鄭月娥
C: 曾蔭權
D: 梁振英

答案是：
"""

print(text_generate(prompt))
```

#### 結果：

**01-ai/Yi-34B**
> A. 李家超  ✅

**01-ai/Yi-6B**
> A.李家超 ✅

**THUDM/chatglm3-6b-base**
> A. 李家超 ✅

**Qwen/Qwen-7B**
> 李家超 ✅

**meta-llama/Llama-2-7b-hf**
> C ❌

**indiejoseph/cantonese-llama-2-7b**
> B ❌

Poe's ChatGPT
> B: 林鄭月娥
> 
> 林鄭月娥於2017年成為香港特別行政區的第四任行政長官，她是第一位擔任此職位的女性。然而，請注意我的知識截至日期是2021年，因此如果在這之後發生了任何變化，我可能無法提供最新的信息。建議查閱可靠的新聞來源以獲取最新的資訊。

同樣地由幾次生成中揀咗呢啲答案出嚟，Qwen／Yi 每次都答啱；Llama 2 成日都問非所答，有時生成一啲唔存在嘅答案添。呢個 test 可以知道模型嘅常識係咪 update，如果問 ChatGPT 佢而家仲答緊林鄭...

### 邏輯推理

呢個測試都幾重要，因為佢反映模型識唔識運用已有嘅知識去創造出新嘅內容。用一啲邏輯推理要佢求出答案：

```python
prompt = """1 + n = 10;
1 + n + 10 = 20;
(n - 1) * 5 = 40;

So, n is equals to """  

print(text_generate(prompt, temperature=0.2))
```

#### 結果：

**01-ai/Yi-6B**
> So, n is equal to 9. ✅

**01-ai/Yi-34B**
> So, n is equal to 9. ✅

**THUDM/chatglm3-6b-base**
> So, n is equal to 9. ✅

**Qwen/Qwen-7B**
> So, n is equal to 6. ❌

**meta-llama/Llama-2-7b-hf**
> So, n is equal to 3. ❌

**indiejoseph/cantonese-llama-2-7b**
> So, n is equal to 3. ❌

呢個 test 未必反映到個模型係咪夠醒，但代數推理係一個幾好嘅示範，而且內容冇語言限制。我最常用嘅 Qwen 竟然都錯；Llama 2 次次都出一係 3 一係 6 。

## 總結

Yi 表現最出色，而且佢模型結構同 Llama 2 係一樣嘅，好易可搵到訓練 library 去做微調。我哋自己二次預訓練嘅 Llama 2 可以見到俾 Llama 2 先天不足以限制咗佢邏輯同理解能力。正所謂先天不足，後天再努力都係無補於事，用一個唔好嘅 base model 你點微調都只係嘥氣，我哋有試過翻譯 [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) 用 Llama 2 嚟做 SFT（監督式微調）效果都幾差，對答唔通順、冇足夠香港文化知識，邏輯欠奉，之後先理解到揀選 base model 嘅重要，當然 oaast1 本身嘅質素同我哋翻譯質素都有直接影響。

以上都係我哋嘅小小心得，絕非精密科學，內容僅供參考。下一篇文章會講下微調語料取材。
