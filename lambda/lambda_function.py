# -*- coding: utf-8 -*-

import logging
import os
import requests
import ask_sdk_core.utils as ask_utils
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-1.5-flash-latest:generateContent"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
SYSTEM_INSTRUCTION = (
    "Você é minha assistente de I.A. Responda de forma concisa e clara, "
    "adequada para ser falada em voz alta pela Alexa. "
    "Limite suas respostas a no máximo 3 parágrafos curtos."
)


def get_history(handler_input):
    session_attr = handler_input.attributes_manager.session_attributes
    return session_attr.get("history", [])


def save_history(handler_input, history):
    session_attr = handler_input.attributes_manager.session_attributes
    session_attr["history"] = history


def send_to_gemini(handler_input, user_message):
    history = get_history(handler_input)

    contents = []
    for entry in history:
        contents.append({
            "role": entry["role"],
            "parts": [{"text": entry["text"]}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "generationConfig": {
            "maxOutputTokens": 300
        }
    }

    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY,
        },
        json=payload,
        timeout=15,
    )
    response.raise_for_status()

    response_data = response.json()
    response_text = (
        response_data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "Não consegui gerar uma resposta.")
    )

    if len(response_text) > 6000:
        response_text = response_text[:6000] + "... Resposta truncada."

    history.append({"role": "user", "text": user_message})
    history.append({"role": "model", "text": response_text})
    save_history(handler_input, history)

    return response_text


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        try:
            text = send_to_gemini(
                handler_input,
                "Olá, me cumprimente brevemente e pergunte como pode me ajudar."
            )
            speak_output = text
        except Exception as e:
            logger.error(e, exc_info=True)
            speak_output = "Erro ao conectar com a assistente. Tente novamente."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class ChatIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("ChatIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        try:
            speak_output = send_to_gemini(handler_input, query)
        except Exception as e:
            logger.error(e, exc_info=True)
            speak_output = "Não obtive uma resposta para sua solicitação."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Alguma outra pergunta?")
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        return (
            handler_input.response_builder
                .speak("Até logo!")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Desculpe, ocorreu um erro. Tente novamente."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChatIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
