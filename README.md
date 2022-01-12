# 你猜我猜 兩小不用猜
# Speech Emotion Prediction

開訊息前常常不知道對方的語氣會是如何，想藉由音訊情緒機器學習判斷來學習如何訓練模型，並把結果應用在LINE Bot上即時回饋

只要加入 LineBot 官方帳號，就可分析語音訊息的情緒哦～

隨時隨地，想猜就猜！

## 馬上體驗
掃一下 QRcode，就可加入LineBot，讓你不用猜兩小

<img src="https://i.imgur.com/Nu3qN2G.jpg" alt="" width="120" style="display:inline-block;">

## 功能說明
1. 加 AudioLineBot 好友後，開啟聊天室即可開始對話
2. 傳送語音訊息，即時收到分析結果，馬上看到訊息的各種情緒預測機率
<p float="left">
<img src="https://i.imgur.com/mivxLpU.jpg" alt="" width="250" style="display:inline-block;">
</p>

3. 顯示各種情緒成分的預測機率，讓你不放過音訊中複雜的情感細節
<p float="left">
<img src="https://i.imgur.com/bfCuc0z.png" alt="" width="500" style="display:inline-block;">
</p>

4. 按下「準」或「不準」按鈕，幫助我們更新更準確的模型吧!
<p float="left">
<img src="https://i.imgur.com/FOqMaIv.jpg" alt="" width="250" style="display:inline-block;">
<img src="https://i.imgur.com/kitAYQj.jpg" alt="" width="250" style="display:inline-block;">
</p>

## 軟體架構圖
<img src="https://i.imgur.com/C3Rscpj.jpg" alt="" width="600" style="display:inline-block;">

## 預測模型 - Model
採用多層感知器(Multi-layer Perceptron)分類模型 --- MLPclassifier

## 特徵萃取 - Feature
由每筆音訊訊號中萃取出3種聲音特徵，當作模型輸入:
1. chroma
2. mfcc
3. mel

## 情緒標籤 - Label
模型輸出6種情緒分類:
1. happy
2. calm
3. angry
4. sad
5. disgust
6. surprised

## 使用工具與套件

- 後端框架：Django==3.2.4
- LINE Bot API: line-bot-sdk==1.19.0
- 機器學習：sklearn==0.0
- 音訊處理：SoundFile==0.10.3.post1
- 音訊特徵萃取：librosa==0.8.1
- 分析結果繪圖：matplotlib
- 音訊轉檔：ffmpeg
- LINE Bot伺服器： ngrok
- 分析結果圖片儲存： imgur