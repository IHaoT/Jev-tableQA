import os
import random
import time

import openai
from openai import OpenAI


# 瞬时故障重试次数。官方原值 3 对中转/公有云端点的 503 不够用。
MAX_RETRIES = int(os.environ.get('GRAPHOTTER_MAX_RETRIES', '6'))


def _backoff(remaining):
    """指数退避 + ±25% 抖动，避免并发线程同时重试。"""
    attempt = MAX_RETRIES - remaining
    return min(2.0 * (2 ** attempt), 30.0) * random.uniform(0.75, 1.25)


def _qwen3_extra_body(model):
    """Qwen3 家族默认开思考模式；GraphOTTER 不需要，显式关闭以节省时间/费用。

    不同端点认的字段不同，两个一起发，多余的会被忽略。
    """
    tail = model.lower().replace('_', '-').rsplit('/', 1)[-1]
    if not tail.startswith('qwen3'):
        return None
    return {'enable_thinking': False, 'thinking': {'type': 'disabled'}}


class ChatGPTTool(object):
    def __init__(self,args):
        self.API_SECRET_KEY = args.key
        self.BASE_URL = args.base_url
        self.model_name = args.model

        self.args = args
        self.extra_body = _qwen3_extra_body(self.model_name)
        # openai 客户端默认 600s，公有云偶发挂起会把整轮实验拖死。
        timeout = float(os.environ.get('GRAPHOTTER_API_TIMEOUT', '120'))
        # chat
        if self.BASE_URL:
            self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL,
                                 timeout=timeout, max_retries=2)
        else:
            self.client = OpenAI(api_key=self.API_SECRET_KEY,
                                 timeout=timeout, max_retries=2)


    def generate(self,prompt,system_instruction = 'You are a helpful AI bot.',isrepeated=0.0,response_mime_type=None):

        if isrepeated > 0.0:
            temperature = self.args.temperature + isrepeated
        else:
            temperature = self.args.temperature
        error = MAX_RETRIES
        resp = None
        kwargs = {}
        if self.extra_body:
            kwargs['extra_body'] = self.extra_body
        while error > 0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": '\n'.join(prompt)}
                    ],
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    # response_format=response_mime_type,
                    **kwargs,
                )

                break
            except openai.RateLimitError as r:
                print('openai限流了',r.__str__())
                error -= 1
                time.sleep(_backoff(error))
            except openai.InternalServerError as r:
                print('openai奔溃了', r.__str__())
                error -= 1
                time.sleep(_backoff(error))
            except openai.APITimeoutError as a:
                print('openai超时', a.__str__())
                error -= 1
                time.sleep(_backoff(error))
            except Exception as r:
                print('openai报错了',r.__str__())
                error -= 1
                time.sleep(2.0)
        if resp is None:
            raise UserWarning('openai 连续失败，已放弃该次调用')
        output = resp.choices[0].message.content
        if output is None:
            raise UserWarning('openai 返回空内容')

        return output
