---
title: 調教廣東話大型語言模型三步曲之二 - 預備微調語料
image: /images/alpaca_with_tank.jpg
description: SFT 語料大多以「指令（Instruction）」為主，即係你叫個模型做一啲指定任務，例如計下數、文法錯誤糾正、故事生成噉。
updated: 2024-02-06 12:05GMT
author: Joseph Cheng
---

![調教廣東話大型語言模型三步曲之二 - 預備微調語料](/images/alpaca_with_tank.jpg)

[上一篇文章](https://hon9kon9ize.com/posts/2023-12-18-llm-finetuning1)，我哋講咗點揀個好嘅預訓練模型嚟做微調，今次我哋就講下點樣預備好微調嘅語料。

### 微調有咩用？

微調有幾種方法而佢哋可以分兩個目的：一）令預訓練模型變成任務模型；二）令輸出更合符人類偏好。
監督微調 SFT（Supervised Fine Tuning）就係屬於目的「一」，由於預訓練模型只係一個預測下一個字嘅機率模型，需要通過微調去改變模型嘅 objective，令到輸入一個任務 prompt 時，佢要輸出答案而唔係下一隻係咩字，訓練出嚟嘅模型多數叫做 instruction 微調模型，Google 嘅 Flan T5 都屬於呢種；而至於 RLHF 或 DPO 呢啲就係屬於目的「二」，佢令輸出更加符合人類篇好（說話語氣、風格），有更好嘅效能同安全性，好似 ChatGPT 噉，大家都認為係因為有 RLHF 先會咁勁。但如果你只想個模式做下啲任務仔嘅話噉呢個步驟係可以慳返嘅。

![Shoggoth with Smiley Face. Courtesy of twitter.com/anthrupad](/images/shoggoth_with_smiley_face.jpg)
*Shoggoth with Smiley Face. 圖片來自 [twitter.com/anthrupad](https://twitter.com/anthrupad)*

### 微調語料類型

SFT 語料大多以「指令（Instruction）」為主，即係你叫個模型做一啲指定任務，例如計下數、文法錯誤糾正、故事生成噉。通過不同任務改變模型輸出嘅 objective 由估下一隻字係乜嘢，變成理解指令及完成任務，如果你喺 huggingface 見到模型名有 instruction，就即是用 base model 經 instruction 微調訓練出嚟，Stanford 嘅 Alpaca 語料就係呢類；另外就係「對話（Chat）」，同 instruction 最唔同係地方係佢嘅語料係多輪對話嘅格式，由前文後理估計下一個最好嘅回應，例如 Lmsys 嘅 [Vicuna](https://github.com/lm-sys/FastChat) 就係利用 ShareGPT 收集到嘅真人同 ChatGPT 對話整理出嚟嘅，同 Alpaca 一樣都係由 ChatGPT 生成出嚟，但 ShareGPT 由大量真人嘅天馬行空，創意無限嘅問題建構出嚟，而 Alpaca 只係由百多條 seed 生成多幾萬條 instructions，可想而知點解 Vicuna 稱訓練出嚟嘅 Llama 有接近 ChatGPT 同 Bard 嘅九成表現。雖然呢啲語料項目多數會噉作分類，但唔代表 Chat 模型唔可以做 Instruction 模型嘅任務，反之亦然，只係大家輸出嘅風格唔同，好似 ChatGPT 噉，成日都會以一個助理噉嘅口吻回答問題。
如果你心目中已經有一個針對嘅 use case，噉你可以搵類似嘅語料作參考，比如：[中醫問診對話](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue)、[法律問答](https://huggingface.co/datasets/LawChat-tw/SFT)或[心理問答](https://huggingface.co/datasets/lsy641/PsyQA)噉。

### 語料生成

了解咗語料種類後，我哋可以開始預備我哋微調嘅訓練語料啦，最常見嘅做法就係直接翻譯現成 dataset 例如 Alpaca，好似呢個 [silk-road/alpaca-data-gpt4-chinese](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese) 噉，佢哋先將 instruction 翻譯，再用 GPT-4 生成出個 output，呢個方法好處係簡單，但翻譯 instruction 時當遇到文法或翻譯嘅任務就會出事：

```
## Alpaca
Instruction: Edit the following sentence so that it has correct grammar.
Input: I were just going to the store
Output: I was just going to the store.

## silk-road/alpaca-data-gpt4-chinese
Instruction: 编辑以下句子，使其具有正确的语法。
Input: 我只是去商店了一下
Output: 我只是去商店了一下
```

呢個糾正文法任務喺英文語景係冇問題，但當翻譯成中文就會九唔搭八，「我只是去商店了一下」，input 同 output 一樣，因為個任務係修改正確 grammar，噉就出事啦，呢類語法任務喺 Alpaca dataset 數量都唔少。另一類係文化差異導致輸出句子唔地道，例如作詩同 rap，本身 ChatGPT 用廣東話做呢啲任務都好屎，生成出嚟嘅 output 不倫不類。因為呢啲因素，最後我哋選擇咗由頭開始生成整個廣東話 Alpaca 語料，我哋先人手翻譯原來嘅 175 條 [seed](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl) ，每條都有 Instruction、Input 同 Output 組成，過程中將英文任務嘅 Input 直接保留只翻譯 Instruction 當係用廣東話指令做英文任務，之後交畀 [Gemini Pro](https://deepmind.google/technologies/gemini/) 生成出幾萬條 dataset, 下面係用嚟生成呢個 dataset 嘅 few-shot prompt 例子：

```
You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a large language model and we will evaluate the large language model for completing the instructions.

Here are the requirements:

1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
2. A large language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in Cantonese.
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 20 tasks:

###

1. Instruction: 畀個句子，輸出所有單字嘅pos標籤。詞性標籤包括形容詞、副詞、連接詞、限定詞、名詞、量詞、介詞、代名詞、動詞。  

呢個係個例子：
約翰好鍾意街尾嗰間藍色屋。

pos 標示結果為： 約翰（名詞）好（副詞）鍾意（動詞）喺（介詞）街尾（名詞）嗰（代名詞）間（量詞）藍色（形容詞）屋（名詞）。

1. Input: 其實我聽唔明你講嘅嘢。
1. Output: 其實（副詞）我（代名詞）聽（動詞）唔（副詞）明（動詞）你（代名詞）講（動詞）嘅嘢（名詞）。

###

2. Instruction:
```

為咗簡化個例子，呢度只用咗一條隨機抽出嚟嘅一組 seed（1. Instruction, 2. Input, 3. Output）放喺 prompt 入面，但我哋正式生成時同 [Alpaca 原代碼](https://github.com/tatsu-lab/stanford_alpaca/tree/main)一樣都係用咗三條（few-shot），用 3 條生多 17 條，之後用 bert 做 embedding 對成個 dataset 做 deduplication，由幾萬條最後剩低萬幾條，心入面諗住  [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) 但同時知道 quality 唔係好好，最後可以做嘅就係參考 [Data-Efficient Instruction Tuning for Alignment” (DEITA)](https://arxiv.org/pdf/2312.15685.pdf) 用 LLM 評分，低過三分就唔要，最後所剩無幾。你可以喺呢度搵到我哋生成出嚟嘅[廣東話 Alpaca 語料](https://huggingface.co/hon9kon9ize/yue-alpaca)同 [script](https://github.com/hon9kon9ize/yue-alpaca)。

以上做法可以應用係唔同嘅生成方法上，好似 [WizradLM](https://arxiv.org/abs/2304.12244)，[DEITA](https://arxiv.org/pdf/2312.15685.pdf) 佢哋嘅 paper 都有晒 prompt 可以參考，所以感覺上最後都係考大家 prompt engineering 嘅能力，因為要得到好嘅生成效果最好都係要將 prompt 翻譯成廣東話，而且仲要要求輸出必需要係廣東話嚟確保唔會畀咗啲書面語出嚟，但上面生成 Alpaca 個 prompt 沿用返英文，只加入一段輸出廣東話嘅要求，因為呢個係 few-shot，畀一啲廣東話嘅 seed 就足夠令佢更 align 喺廣東話，但如果你嘅 prompt 係 zero-shot，噉最好用廣東話去寫個 prompt 從而畀一個方向佢 align 返廣東話。

### 總結

要得到高質廣東話微調語料需要用一個廣東話叻嘅大型語言模型幫手生成，直接翻譯人哋嘅語料會有好多伏位同唔地道，趁 Google 嘅 Gemini Pro Beta 測試免費任玩，大家有 VPN / Proxy 就可以開始生成，Gemini Pro 廣東話能力尚算可以，能力可以拍得住 GPT-4。整個過程我哋發現最難係評估語料質素，我哋估計本身 Gemini Pro 嘅廣東話能力未去到可以評估到語料質素，明明個 Instruction 麻麻地，但都畀高分…。我哋從好多 papers 知道質比量重要，[WizradLM](https://arxiv.org/abs/2304.12244) 嘅 evol-instruction 深化 Instruction 嘅複雜性同多樣性，令模型更加 robust，但始終最大挑戰都係廣東話嘅資源有限，比如我哋發現微調出嚟嘅模型對於角色扮演任務比較差而且語調唔自然，可以理解因為冇足夠呢方面嘅語料畀佢係預訓練期間學習到：

```
Human: 你要扮成我女朋友 Karen 嚟答我問題，Karen 係一個好靚嘅女仔，佢有一顆善良嘅心。

Karen，我令日好忙，而家先有時間打畀你，你令日點呀？

Assistant: 我令日都幾好呀，多謝你嘅關心。我喺度做緊嘢，重喺度整緊嘢食。我今日返咗工之後，就去咗間餐廳度食飯，重喺度買咗啲嘢食返嚟。我重喺度睇緊電視，重喺度聽緊音樂。我今日好開心，你呢？
```

可以見到一啲都唔入戲...

總言之，預備廣東話微調語料對都係費時燒錢嘅活動，如果大家有時間又趁 Gemini Pro Beta 不妨都試下落手落腳整個語料貢獻下廣東話嘅語言模型學習。