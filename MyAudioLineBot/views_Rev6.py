from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
 
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, StickerSendMessage, ImageSendMessage, TemplateSendMessage, FlexSendMessage, ButtonsTemplate, MessageTemplateAction, URITemplateAction 
import json

# import python packages for speech emotion recognition
from pydub import AudioSegment
import speech_recognition as sr
import librosa
import soundfile
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pyimgur

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

sticker_dict = dict()
sticker_dict['calm'] = {1: {'package_id': '446', 'sticker_id': '1998'},
                        2: {'package_id': '446', 'sticker_id': '2009'},
                        3: {'package_id': '446', 'sticker_id': '2015'},
                        4: {'package_id': '446', 'sticker_id': '2010'}}
sticker_dict['happy'] = {1: {'package_id': '446', 'sticker_id': '1991'},
                        2: {'package_id': '446', 'sticker_id': '1988'},
                        3: {'package_id': '446', 'sticker_id': '1992'},
                        4: {'package_id': '446', 'sticker_id': '1990'}}
sticker_dict['sad'] = {1: {'package_id': '446', 'sticker_id': '2007'},
                        2: {'package_id': '446', 'sticker_id': '2020'},
                        3: {'package_id': '446', 'sticker_id': '2022'},
                        4: {'package_id': '446', 'sticker_id': '2023'}}
sticker_dict['angry'] = {1: {'package_id': '446', 'sticker_id': '2019'},
                        2: {'package_id': '446', 'sticker_id': '2004'},
                        3: {'package_id': '6136', 'sticker_id': '10551379'},
                        4: {'package_id': '11537', 'sticker_id': '52002773'}}
sticker_dict['disgust'] = {1: {'package_id': '11538', 'sticker_id': '51626526'},
                        2: {'package_id': '446', 'sticker_id': '2006'},
                        3: {'package_id': '446', 'sticker_id': '2025'},
                        4: {'package_id': '11537', 'sticker_id': '52002760'}}
sticker_dict['surprised'] = {1: {'package_id': '446', 'sticker_id': '2024'},
                        2: {'package_id': '446', 'sticker_id': '2011'},
                        3: {'package_id': '446', 'sticker_id': '2017'},
                        4: {'package_id': '446', 'sticker_id': '2016'}}
def give_me_a_sticker(mood, sticker_dict):
    index = randint(1, 4)
    p_id = sticker_dict[mood][index]["package_id"]
    s_id = sticker_dict[mood][index]["sticker_id"]

    sticker_message = StickerSendMessage(
        package_id=p_id,
        sticker_id=s_id
        )
    return sticker_message

def give_me_a_image(y_pred_prob, model, user_id):
    y_pred_prob_0 = np.array(100*y_pred_prob[0])

    plt.figure(dpi=120)
    plt.bar(model.classes_,
            y_pred_prob_0, 
            width=0.5, 
            bottom=None, 
            align='center', 
            color=['lightsteelblue', 
                   'cornflowerblue', 
                   'royalblue', 
                   'midnightblue', 
                   'navy', 
                   'darkblue'],)
    plt.xticks(rotation='vertical')
    plt.ylabel('Probability (%)')
    plt.xticks(rotation=45)
    # plt.axhline(y=50, c="r", ls="--", lw=1)
    plt.grid(axis='y', linestyle='dotted', lw=1)
    plt.tight_layout()
    plt.savefig("./Image/" + user_id + ".png")

    CLIENT_ID = "a11f77bf955274f"
    # A Filepath to an image on your computer"
    PATH = "./Image/" + user_id + ".png" 
    title = user_id + "emotion components"
    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(PATH, title=title)
    url = uploaded_image.link

    image_message = ImageSendMessage(
        original_content_url=url,
        preview_image_url=url
        )
    return image_message


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
            events = parser.parse(body, signature)  # ???????????????
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
 
        for event in events:
            if isinstance(event, MessageEvent):  # ?????????????????????
                message=[]
                if event.message.type == "text": # ?????????????????????
                    print(event.message.text)

                    user_id = event.source.user_id
                    print(event.source.user_id)

                    line_bot_api.reply_message( # ??????????????????????????????
                        event.reply_token,
                        TextSendMessage(text="??????~?????????????????????")
                    )
                
                elif event.message.type == "audio": # ?????????????????????
                    
                    user_id = event.source.user_id
                    print(event.source.user_id)

                    path = "./Audio/" + user_id + ".wav"
                    audio_content = line_bot_api.get_message_content(event.message.id)
                    with open(path, 'wb') as fd:
                        for chunk in audio_content.iter_content():
                            fd.write(chunk)
                    fd.close()

                    # ???wav???
                    # ????????????????????????ffmpeg.exe??????
                    AudioSegment.converter = 'C:\\Users\\user\\ffmpeg\\bin\\ffmpeg.exe'
                    sound = AudioSegment.from_file_using_temporary_files(path)
                    path = os.path.splitext(path)[0]+'.wav'
                    sound.export(path, format="wav")

                     #???????????????????????????
                    r = sr.Recognizer()
                    sound = AudioSegment.from_file_using_temporary_files(path)
                    with sr.AudioFile(path) as source:
                        r.adjust_for_ambient_noise(source, duration=0.1)
                        audio = r.record(source)
                    # ??????????????????????????????
                    # ??????: language='zh-TW'
                    # English: language='en-US'
                    try:
                        text = "????????????: " + r.recognize_google(audio, language='zh-TW')
                    except sr.UnknownValueError:
                        text = "???????????????????????????????????????"
                    except sr.RequestError as e:
                        text = "????????????{0}".format(e)
                    print(text)
                    # ?????????????????????????????????
                    message.append(TextSendMessage(text=text))


                    # load the MLP model & its scaler
                    # model = pickle.load(open('C:\\Users\\user\\best_model.pkl', 'rb'))
                    model = pickle.load(open('C:\\Users\\user\\AD_Project_LineBot_use_model.pkl', 'rb'))
                    scaler = pickle.load(open('C:\\Users\\user\\AD_Project_LineBot_use_scaler.pkl', 'rb'))

                    # DataFlair - Load the data and extract features for each sound file
                    x = []
                    for file in glob.glob("C:\\Users\\user\\MySecondDjango\\Audio\\" + user_id +".wav"):
                        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
                        x.append(feature)
                    x_scaled = scaler.transform(x)
                    y_pred = model.predict(x_scaled)
                    y_pred_prob = model.predict_proba(x_scaled)
                    print(y_pred_prob)
                    
                    for y_hat in y_pred:
                        y_reply = str(y_hat)
                    text = "????????????: " + y_reply
                    print(text)
                    # ?????????????????????????????????
                    message.append(TextSendMessage(text=text))
                    # ???????????????????????????????????????????????????
                    message.append(give_me_a_sticker(y_reply, sticker_dict))
                    # ??????????????????????????????????????????????????????
                    message.append(give_me_a_image(y_pred_prob, model, user_id))
                    # ????????????????????????
                    line_bot_api.reply_message(event.reply_token, message)

        return HttpResponse()
    else:
        return HttpResponseBadRequest()
