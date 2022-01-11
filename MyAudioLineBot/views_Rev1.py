from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
 
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, TemplateSendMessage, FlexSendMessage, ButtonsTemplate, MessageTemplateAction, URITemplateAction 
import json
import os

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)
 
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
                if event.message.type == "text":
                    print(event.message.text)

                    user_id = event.source.user_id
                    print(event.source.user_id)

                    line_bot_api.reply_message( # 回復傳入的訊息文字
                        event.reply_token,
                        TextSendMessage(
                            text=event.message.text,
                        )
                    )
                
                elif event.message.type == "audio":
                    message.append(TextSendMessage(text='聲音訊息'))
                    user_id = event.source.user_id
                    print(event.source.user_id)
                    path = "./audio/" + user_id + ".wav"
                    audio_content = line_bot_api.get_message_content(event.message.id)
                    with open(path, 'wb') as fd:
                        for chunk in audio_content.iter_content():
                            fd.write(chunk)
                    fd.close()
                    line_bot_api.reply_message(event.reply_token,message)
        return HttpResponse()
    else:
        return HttpResponseBadRequest()