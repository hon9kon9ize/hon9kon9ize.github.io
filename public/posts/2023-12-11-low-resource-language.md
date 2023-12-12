---
title: 大語言模型點樣令廣東話走出低資源語言嘅惡咒
image: /images/low-resource-lang.jpg
description: 定義上指該語言可以取手可得到嘅人工標籤語料非常稀缺，即是如果想做一啲機器學習或自然語言（NLP）任務時好難隨手搵到資源。
updated: 2023-12-11 12:05GMT
author: Joseph Cheng
---

![大語言模型點樣令廣東話走出低資源語言嘅惡咒](/images/low-resource-lang.jpg)

## 低資源語言（Low-resource language）

定義上指該語言可以取手可得到嘅人工標籤語料非常稀缺，即是如果想做一啲機器學習或自然語言（NLP）任務時好難隨手搵到資源。一個簡單例子等大家掌握一下廣東話有幾低資源，一般 NLP 通常都需要建構一個統計模型嚟了解下自己手上嘅 dataset，比如係入面某個字出現咗幾多次，或者係某兩個字一齊出現過幾多次（2-grams），呢個時候我哋可以對照一下自己個 dataset 同其他 dataset 字詞分佈嘅咪相近，作一個參考，但廣東話最大型嘅 dataset 就只有[粵維基百科](https://zh-yue.wikipedia.org/)，字數大約係三千五佰萬，檔案大小唔到 60mb，對比起[中文維基百科](https://zh.wikipedia.org/)，字數大約十億字，檔案大少接近 600mb，即是相差過佰倍之多。噉即係有咩影響呢？機器學習係一個講概率嘅學習方式，一個用詞出現得越多就越有機會係一個正確嘅詞語，亦可以由上文下理中統計其他字嘅共同出現頻率嚟得出該詞語意思。從以下截圖可以睇到，常用嘅廣東話字詞喺粵維基中嘅字頻：

```python
from datasets import load_dataset

df = load_dataset('indiejoseph/wikipedia-zh-yue-filtered', split='train').to_pandas()

words = ['戇居', '𧥺水', '瞌眼瞓', '搵笨', '捱夜', '打乞嚏', '擤鼻涕']

for word in words:
    print(word, df['text'].str.count(word).sum())

#戇居 5
#𧥺水 0
#瞌眼瞓 14
#搵笨 11
#捱夜 3
#打乞嚏 4
#擤鼻涕 2
```

因為粵維基係一個知識庫，內容比較少出現一啲口語常會用到嘅字。

## 異體字、錯別字、俗寫字...

我唔係語言專家以下嘅嘢可能有錯（帶定頭盔），但想講下整理廣東話語料時發現到嘅有趣嘢。我經常同朋友傾返細個上中文堂學嘅字，同大個咗發現睇到嘅字有啲唔同，例如「裏」字，因為網絡年代香港人多數都用台灣嘅輸入法，所以不知不覺就跟咗台灣標準「裡」，仲有「為」同「爲」、「群」和「羣」噉，仲有不少呢啲所謂嘅俗體字，噉究竟個機器學到啲咩呢？佢大概都會發現呢啲字係同義字因為都喺差不多嘅字附近出現，但如果想語言模型輸出更統一時，呢啲差異一定會有影響。

```python
words = [
	'為', '爲',
	'群', '羣',
	'裏', '裡',
	'粧', '妝',
	'歎', '嘆'
]

for word in words:
    print(word, df['text'].str.count(word).sum())

#為 81520 vs 爲 7440
#群 7001 vs 羣 1320
#裏 5228 vs 裡 1152
#粧 111 vs妝 318
#歎 91 vs 嘆 211
```

[俗寫字](https://zh-yue.wikipedia.org/wiki/%E4%BF%97%E5%AD%97)定義係只要多人用嘅就係俗寫字，一個極端例子，「d」即是「啲」，雖然唔多但喺粵維基都有佢嘅踪影，如果喺 Common Crawl 度就可能係會佔大多數，廣東話俗寫字比其他語言多好多，因為你識講但都未必識寫，比如「冚唪唥」、「捩咁棄」、「亂噏」噉，大家會用同音字代替。

```python
lines_df = df['text'].str.split('\n').explode()

print('\n'.join(lines_df[lines_df.str.contains(r'呢d[^a-zA-Z0-9]')].values))

# - 呢d就係人地對日本人嘅兩個誤解
# - 2017年：《Bank友講呢d -老老竇竇做問卷》飾演投資顧問
```

本身都低資源㗎喇，仲要有一大堆異體字、錯別字，就算畀 Common Crawl 你篩晒啲廣東話出嚟，嗰啲俗寫字都會令你頭都大埋，需要花好多資源去糾正。

## 廣東話成為低資源語言嘅原因

廣東話成為低資源語言嘅原因有好多，除咗上面所講嘅俗寫字，仲有一定程度係因為文化上同學術上嘅影響。我聽過有台灣人覺得香港人好似天生識兩種語言，即是我哋所謂嘅「口語」同「書面語」，我哋自細開始就有兩個唔同嘅語言系統，作文一定係書面語，但冇人會用書面語講嘢，所以你會發現叫你睇廣東話文章可能會比書面語讀得慢。所有書、電視字幕、報紙同網上文章都以書面語為主流；另外學術上，好少聽到有課程係會教人寫廣東話，當然出面一定有學講廣東話嘅課程，但廣東話冇一個標準寫法，可能個個都有自己用字準則。過去都有唔少有關廣東話嘅學術研究，好多都着重嘅會話上，最多人知嘅應該係廿幾年前嘅 [hkcancor](https://github.com/fcbond/hkcancor) 一萬句口語錄音同標籤語料，另外最出色就係近年嘅 [Unsupervised Mandarin-Cantonese Machine Translation](https://arxiv.org/abs/2301.03971#:~:text=Advancements%20in%20unsupervised%20machine%20translation,abundance%20of%20parallel%20data%20available.)，佢哋係爬咗網絡上唔同嘅廣東話文章再由逐句逐句喺簡體中文句子中發掘相似語義句子，最出一萬條翻譯對。另外仲有 pyCantonese 列出一堆嘅可用 [dataset](https://pycantonese.org/data.html)。如果將所有可用嘅語料加埋一齊，檔案大小都仲係離 1gb 差好遠，規模上始終仲係太細。

## 低資源噉又點啫？

對於細公司冇咩資源嚟講，想做廣東話方面嘅產品開發真係舉步為艱，但對於大公司嚟講就緊係唔係問題。好似 Apple 噉，Siri 都識講廣東話，語音辨識都支援廣東話，只要有商業動機，佢哋就有資源去做；Open AI 嘅 ChatGPT 都會講廣東話啦，但有玩開就知，就算 GPT-4 佢都係唔流利，仲成日混埋啲簡體字添。

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "用廣東話講個笑話聽下先！"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Output:
###
有一日，一只雞跟一隻兔兩兄弟把車開去自己屋企。突然間車就壞咗。佢哋話，「我哋講咩先好喎」，雞就話，「你瞓覺，我哋等下開始講笑話。」
兔擺擺手話，「咁我一出門就會累瞓，你先講笑話。」
於是雞就笑著講，「你認唔認得呢隻兔仔，原來唔知得車有冷氣，仲要用紙扇自己！」聽後，兔笑到人仰馬翻。
###
```

因為 GPT 呢啲大語言模型（ Large Language Model）都係由網絡上搵到嘅文字訓練出嚟，越多高質素嘅文字就表現得越好，加上高質嘅正體中文同簡體中文比廣東話要多好多倍，都會影響到 LLM 輸出時點決定出下一隻字嘅機率，即是佢比較識講官話，所以廣東話唔流利噉解。但有一日佢一定會更流利，只要佢見到廣東話有市場嘅話。

所以做廣東話 NLP 嘅人都係深耕細作，冇資源出品就自然少，但有咗不同嘅 LLMs 希望可以改善呢個問題。

## 「修正錯別字」同「廣東話句子篩選」

最快獲得大量廣東話句子嘅方法就緊係爬晒網絡上嘅廣東話文章啦，我哋可以去 [Common Crawl](https://commoncrawl.org/) 下載最近嘅 snapshot 從中發掘，但要處理數十 TiB 嘅 raw 文字真係唔容易，所以我哋可以用其他人處理好嘅 dataset 從中再篩選，Meta 嘅 [CCNet 100](https://data.statmt.org/cc-100/)、Google 嘅 [C4](https://www.tensorflow.org/datasets/catalog/c4) 都係一啲處理好（dedup, language detect）嘅語料，我哋可以用 HuggingFace 上面人哋喺呢啲語料篩好嘅正體中文 datasets 去再篩廣東話出嚟。

```python
from datasets import load_dataset

c4_df = load_dataset('erhwenkuo/c4-chinese-zhtw', split='train').to_pandas()
cc100_df = load_dataset('zetavg/CC-100-zh-Hant', split='train').to_pandas()

print('C4:', len(c4_df))
print('CC-100:', len(cc100_df))

# Before filtered:
# C4: 2967556 # 2.96M rows
# CC-100: 85165683 # 85M rows
```

再用 [CanCLID](https://github.com/CanCLID) 開發嘅[粵文分類篩選器](https://github.com/CanCLID/canto-filter) 可以得到一柞廣東話句子：

```python
from cantofilter import judge

c4_df['lang'] = c4_df['text'].apply(lambda x: judge(x))
cc100_df['lang'] = cc100_df['line'].apply(lambda x: judge(x))

print('C4:', c4_df[c4_df['lang'] == 'cantonese'])
print('CC-100:', cc100_df[cc100_df['lang'] == 'cantonese'])

# After filtered:
# C4: 6633 # 6.6k rows
# CC-100: 233148 # 23.3k rows
```

但問題嚟喇！篩完冇咗得 2% 左右剩，真係得咁少？其實因為好多俗字廣東話句子都被篩走埋：

```python
text = "做on9野引人笑既_on9仔？"

judge(text)

# Output: neutral
```

輸出係「neutral」即是可能係廣東話或者係官話，呢個例子因為「野」、「既」同「on9」都係借咗同音字，而呢啲好多時都係喺 Common Crawl 中佔大比數，原因係網上爬到嘅，好多時都係啲食評、社交媒體、部落格等，呢類文字一般都係冇咁嚴格，所以要先修正好錯別字、俗寫字噉會可以再得到更多好嘅廣東話句子， [CanCLID](https://github.com/CanCLID) 有開發一個 rule base [錯別字修正器](https://github.com/CanCLID/typo-corrector)，一般 [語氣詞](https://jyutping.org/blog/particles/) 都可以幫到啲手：

```python
from typo_corrector.rules import apply_contextual_rules
import re

...

def correct(line):
	stripped_line = line.strip()
	fixed = fix_regular_typo(stripped_line)
	fixed = apply_contextual_rules(fixed)
	return fixed
	
print(correct('做on9野引人笑既_on9仔？'))
# Output: 做戇𨳊野引人笑嘅 _ 戇𨳊仔？ # 「野」仲係錯，因為冇呢個 rule
print(correct('我家姐係我最好既親人'))
# Output: 我家即係我最好嘅親人 # 誤將「姐係」當係錯別字「即係」
```

rule base 不分上文下理，所以會出現一啲錯判情況，文字配搭千變萬化 rule base 難以幾個 rules 就解決晒所有問題。呢個時候我哋可以用 bert 呢類細 language model 去判斷錯別字嘅可能性，簡單噉比兩個字佢嚟判斷邊個機會大啲：

```python
from transformers import pipeline

pipe = pipeline('fill-mask', model='indiejoseph/bert-base-cantonese', device=0)

# 「比」 vs 「畀」
# Input: 之後佢會copy poassport就比張飛仔同帶左我去量血壓、探熱。
print(pipe('之後佢會copy poassport就[MASK]張飛仔同帶左我去量血壓、探熱。', targets=['比', '畀']))
# Output: 畀(score: 0.031) vs 比(score: 0.001)

# Input: 佢一定比你靚得多啦！
print(pipe('佢一定[MASK]你靚得多啦！', targets=['比', '畀']))
# Output: 畀(score: 0.00) vs 比(score: 0.9)

# Input: 我家姐係我最好嘅親人
print(pipe('我家[MASK]係我最好嘅親人', targets=['姐', '即']))
# Output: 姐(score: 0.61) vs 即(score: 0.00)

# Input: 
print(pipe('畀我錫一[MASK]？', targets=['吓', '下']))
# Output: 吓(score: 0.21) vs 下(score: 0.12)

# Input: 
print(pipe('喂？[MASK]！我有冇聽錯？', targets=['吓', '下']))
# Output: 吓(score: 0.01) vs 下(score: 0.00)

# Input: 呢D嘢，唔到我話事
print(pipe('呢[MASK]嘢，唔到我話事', targets=['D', '啲']))
# Output: D(score: 0.00) vs 啲(score: 0.42)

```

可以見到，如果我哋能夠先假定一柞錯別字同相應嘅修正，我哋就可以用 bert 作為機率參考，因為 bert 會由上文下理嚟分析，所以誤判機會低。呢個 bert base model 我用咗 wikipedia 同一柞 c4 嘅語料去訓練，人手修正咗少部分錯別字，佢嘅目的雖然唔係主要用嚟做錯別字判斷，但都有一定程度嘅幫助。

由上面例子中可見，多得近年 LLMs 預訓練對語料量嘅龐大需求，多咗好多處理好嘅中文語料，雖然唔係針對廣東話，但起碼我哋可以借台灣人處理嘅繁中語料從中攞到便利。

## 借助 LLMs 生成廣東話語料

因最近喺度訓練廣東話翻譯模型，當中大量借助 GPT-3.5 同 Palm 2 嚟生成翻譯語料，之前都講咗 GPT-4 嘅廣東話都係麻麻地，但佢哋由廣東話翻譯去其他語言效果都還可以：

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "翻譯以下句子做普通話"},
        {"role": "user", "content": "噉你而家即係想點吖？"}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Output: 那你现在想要怎么样呢？
```

噉樣，我哋可以用一柞廣東話，畀 GPT 生成翻譯對。

另外，我仲發現 Palm 2 嘅廣東話係咁多 LLMs 之中最好：

```python
import vertexai

from vertexai.language_models import TextGenerationModel
  

vertexai.init(project="arbot-68f03", location="us-central1")
parameters = {
	"candidate_count": 1,
	"max_output_tokens": 1024,
	"temperature": 0.2,
	"top_p": 0.8,
	"top_k": 40
}

model = TextGenerationModel.from_pretrained("text-bison")

response = model.predict("""Translate the following Chinese Sentences to Cantonese(廣東話口語)

我很抱歉您现在感到压力和不知所措。或许您可以尝试以下一些方法来帮助释放压力和减轻您的不安

Output:""",
	**parameters
)

print(f"Response from Model: {response.text}")

# Response from Model: 我好抱歉你而家覺得好大壓力又唔知點算。或者你可以嘗試吓以下一啲方法嚟幫助釋放壓力同舒緩你嘅不安
```

所以我用齊兩種方法嚟生成翻譯訓練語料。首先我會用種子詞畀 GPT-3.5 去生成 10 段句子：

```python
seed_terms = {
	"打邊爐": "食火锅"
	"死仔包": "臭小子"
	"水皮": "技术差"
	"及格分數": "及格线"
	"濕星嘢": "琐事"
	...
}

# generate 10 sentences
output = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "用「{}」生成十段句子".format(seed_terms["解气"])}
    ],
)

print(output)

###
1. 她长久以来在工作中受到了不公平对待，但最终通过努力和坚持，成功地解气了。
2. 经过一番努力和奋斗，他终于在比赛中战胜了一直欺负他的对手，感到非常解气。
3. 长时间被别人忽视和轻视，但他通过展现自己的才能和价值，最终解气地赢得了大家的认可。
4. 曾经遭受不公正的对待，但通过法律的维权，她成功地解气了。
5. 在面对生活中的种种挑战和困难后，他通过坚持不懈的努力，成功地解气了。
6. 经历了一段痛苦的过去，他通过学习和成长，最终解气地走出了阴影。
7. 经历了一次失败后，他重新振作起来，通过取得成功，解气地证明了自己的能力。
8. 在遭受背叛和欺骗后，她通过追求自己的幸福，最终解气地找到了更好的人生。
9. 经历了一番努力后，他终于实现了自己的目标，感到非常解气。
10. 长期被人质疑和贬低，但他通过出色的表现，最终解气地证明了自己的价值。
###
```

之後用 Palm2 將佢哋翻譯做廣東話，留意下面，我用咗 few-shot，同 in-context learning 嘅方法令輸出效果更加好：

```python
output = model.predict("""  
As a Cantonese translator, your job is to accurately and locally translate the given input into Cantonese(廣東話口語).
 
Glossary:
噉：廣州音[gam2敢]
(解)①指示代詞，這樣，那樣。(例)噉就好嘞。[這樣就好了。] || 呢件事大家都噉睇。[這件事大家都那樣看。]②助詞，似的。(例)苦瓜噉面。[像苦瓜似的臉，比喩愁眉苦臉] || 打鑼噉聲。[形容聲音大，像打鑼似的。]③助詞，地。(例)細細聲噉講。

咁：廣州音[gam3禁]
(解)①這麼；那麼。(例)咁多就夠嘞。[這麼多就夠了] || 枝竹竿要咁長先啱用。[竹竿要這麼長才合用。]②多麼。(例)咁靚[多麼漂亮] || 天氣鬼咁熱。[天氣多麼熱。]

畀：廣州音[bei2比]
(解)①給(例)畀本書我。[給我一本書。] || 你話畀佢知。[你告訴他/她。]②介詞。被；受；讓；用。(例)畀人睇見。[被人看見。] || 畀嗰個人煩咗成日。[受那個人干擾了整天。] || 畀佢入嚟。[讓他/她進來。] || 畀墨水筆寫。[用鋼筆寫。]③連詞，常與「着」連用，表示假設。(例)畀着你係主角，你會點做？[假設你是主角，你會怎做？]

俾：廣州音[bei2比]
(解)使。(例)俾便考查。 || 唔好俾度門關埋。[使門不能關上。] 

Input:
初二数学，给定一组数据的方差，很多人看完题直接放弃，你会吗？
突然停电了，大家都有点惊慌失措地在黑暗中摸索着找寻手机和手电筒。
小男孩一脸淘气地笑着，调皮地捏了一下姐姐的脸。
她长久以来在工作中受到了不公平对待，但最终通过努力和坚持，成功地解气了。
经过一番努力和奋斗，他终于在比赛中战胜了一直欺负他的对手，感到非常解气。
长时间被别人忽视和轻视，但他通过展现自己的才能和价值，最终解气地赢得了大家的认可。
曾经遭受不公正的对待，但通过法律的维权，她成功地解气了。
在面对生活中的种种挑战和困难后，他通过坚持不懈的努力，成功地解气了。
经历了一段痛苦的过去，他通过学习和成长，最终解气地走出了阴影。
经历了一次失败后，他重新振作起来，通过取得成功，解气地证明了自己的能力。
在遭受背叛和欺骗后，她通过追求自己的幸福，最终解气地找到了更好的人生。
经历了一番努力后，他终于实现了自己的目标，感到非常解气。
长期被人质疑和贬低，但他通过出色的表现，最终解气地证明了自己的价值。 

Output:
初二數學，畀一組數據嘅方差，好多人睇完題就放棄，你識唔識做？
突然停電，大家都慌失失噉摸黑搵電話電筒。
個男仔成個百厭樣噉笑住，仲好曳噉搣咗佢家姐塊面一下。
""",
	**parameters
)

# Output:
###
佢喺工作入面長久噉受到唔公平嘅對待，但最後通過努力同堅持，成功噉出咗啖氣。
經過一番努力同奮鬥，佢終於喺比賽入面贏咗成日蝦佢嗰個對手，覺得好爽。
長時間畀人忽視同睇唔起，但佢通過展現自己嘅才能同價值，最後爽噉贏得咗大家嘅認可。
曾經受到唔公平嘅對待，但通過法律嘅維權，佢成功噉出咗啖氣。
喺面對生活入面嘅種種挑戰同困難之後，佢通過堅持不懈嘅努力，成功噉出咗啖氣。
經歷咗一段痛苦嘅過去，佢通過學習同成長，最後爽噉行出咗陰霾。
經歷咗一次失敗之後，佢重新振作返，通過取得成功，爽噉證明咗自己嘅能力。
喺受到背叛同欺騙之後，佢通過追求自己嘅幸福，最後爽噉搵到咗更好嘅人生。
經歷咗一番努力之後，佢終於實現咗自己嘅目標，覺得好爽。
長期畀人質疑同貶低，但佢通過出色嘅表現，最後爽噉證明咗自己嘅價值。
###
```

出嚟效果唔錯，之後人手再做一次簡單校對就用得，我有大約 4 千對 seed terms 每對生成十段句子，最後獲得 4 萬對翻譯對，用一成嚟做 test set。最後再用入面嘅廣東話句子做畀 GPT 做 back translate，噉就可以又多一部嘅 dataset 啦！雖然 Palm 2 廣東話出色，但比起 GPT 佢真係蠢好多，成日唔聽指令，又成日輸出粗口...，希望之後會有改進。

以上係一個簡單例子，我哋可以見到一個效應，當 LLMs 生成嘅語料去到一個咁上下嘅質素時，價錢同質量就會大幅改變，好平就可以買到一柞好語料，噉廣東話就唔會再係低資源語音啦。

嚟緊我哋會訓練一個廣東話嘅 LLM 嚟生成更多好嘅語料，希望可以為廣東話社群做少少貢獻。