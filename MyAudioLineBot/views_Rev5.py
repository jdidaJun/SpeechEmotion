from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
 
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, StickerSendMessage, TemplateSendMessage, FlexSendMessage, ButtonsTemplate, MessageTemplateAction, URITemplateAction 
import json

# import python packages for speech emotion recognition
from pydub import AudioSegment
import speech_recognition as sr
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result

def give_me_a_sticker(mood):
    if mood == 'calm':
        sticker_message = StickerSendMessage(
            package_id='446',
            sticker_id='1998'
        )
    elif mood == 'happy':
        sticker_message = StickerSendMessage(
            package_id='446',
            sticker_id='1991'
        )
    elif mood == 'sad':
        sticker_message = StickerSendMessage(
            package_id='446',
            sticker_id='2007'
        )
    elif mood == 'angry':
        sticker_message = StickerSendMessage(
            package_id='446',
            sticker_id='2019'
        )
    elif mood == 'disgust':
        sticker_message = StickerSendMessage(
            package_id='11538',
            sticker_id='51626526'
        )
    elif mood == 'surprised':
        sticker_message = StickerSendMessage(
            package_id='446',
            sticker_id='2024'
        )
    return sticker_message

# LineBot operation
line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

# LineBot main function
@csrf_exempt
def callback(request):
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
 
        try:
            events = parser.parse(body, signature)  # 傳入的事件
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
 
        for event in events:
            if isinstance(event, MessageEvent):  # 如果有訊息事件
                message=[]
                if event.message.type == "text": # 如果是文字訊息
                    print(event.message.text)

                    user_id = event.source.user_id
                    print(event.source.user_id)

                    line_bot_api.reply_message( # 請使用者輸入語音訊息
                        event.reply_token,
                        TextSendMessage(text="你好~請輸入語音訊息")
                    )
                
                elif event.message.type == "audio": # 如果是語音訊息
                    
                    user_id = event.source.user_id
                    print(event.source.user_id)

                    path = "./Audio/" + user_id + ".wav"
                    audio_content = line_bot_api.get_message_content(event.message.id)
                    with open(path, 'wb') as fd:
                        for chunk in audio_content.iter_content():
                            fd.write(chunk)
                    fd.close()

                    # 轉wav檔
                    # 輸入自己電腦內的ffmpeg.exe路徑
                    AudioSegment.converter = 'C:\\Users\\user\\ffmpeg\\bin\\ffmpeg.exe'
                    sound = AudioSegment.from_file_using_temporary_files(path)
                    path = os.path.splitext(path)[0]+'.wav'
                    sound.export(path, format="wav")

                     #進行語音轉文字處理
                    r = sr.Recognizer()
                    sound = AudioSegment.from_file_using_temporary_files(path)
                    with sr.AudioFile(path) as source:
                        r.adjust_for_ambient_noise(source, duration=0.1)
                        audio = r.record(source)
                    # 設定要以什麼文字轉換
                    # 繁中: language='zh-TW'
                    # English: language='en-US'
                    try:
                        text = "訊息內容: " + r.recognize_google(audio, language='zh-TW')
                    except sr.UnknownValueError:
                        text = "無法辨識語音訊息內文字內容"
                    except sr.RequestError as e:
                        text = "無法翻譯{0}".format(e)
                    print(text)
                    #將轉換的文字回傳給用戶
                    message.append(TextSendMessage(text=text))


                    # load the MLP model
                    # model = pickle.load(open('C:\\Users\\user\\AD_Project_Test_13_model.pkl', 'rb'))
                    model = pickle.load(open('C:\\Users\\user\\best_model.pkl', 'rb'))

                    # DataFlair - Load the data and extract features for each sound file
                    x = []
                    for file in glob.glob("C:\\Users\\user\\MySecondDjango\\Audio\\" + user_id +".wav"):
                        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
                        x.append(feature)
                    y_pred = model.predict(x)

                    for y_hat in y_pred:
                        y_reply = str(y_hat)
                    text = "訊息情緒: " + y_reply
                    print(text)
                    message.append(TextSendMessage(text=text))
                    message.append(give_me_a_sticker(y_reply))
                    line_bot_api.reply_message(event.reply_token, message)

        return HttpResponse()
    else:
        return HttpResponseBadRequest()
