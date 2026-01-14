# -*- coding: utf-8 -*-
import requests
import json
import config

class CompletionExecutor:
    def __init__(self, hcx_type, call_type):
        host = ''
        api_key = config.HCX_API_KEY
        api_key_primary_val = config.HCX_APIGW_KEY
        request_id = config.HCX_REQUEST_ID

        if hcx_type == 'dash' and call_type == 'stream':
            host = config.HCX_HOST_DASH
        elif hcx_type == 'dash' and call_type == 'json' :
             host = config.HCX_HOST_DASH_GW
        elif hcx_type == '003' and call_type == 'stream' :
             host = config.HCX_HOST
        else :
             host = config.HCX_HOST_GW 

        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        request_data = {
            'messages': completion_request,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.5,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }
    
        with requests.post(self._host,
                            headers=headers, json=request_data, stream=True) as r:
                is_result = False
                for line in r.iter_lines():
                    if line:
                        content = line.decode("utf-8")
                        if is_result:
                            res = json.loads(content[5:])
                            return res["message"]["content"]

                        if content == "event:result":
                            is_result = True
    
    def execute_json(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
        }

        request_data = {
            'messages': completion_request,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.5,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }
    
        with requests.post(self._host,
                            headers=headers, json=request_data, stream=True) as r:
                return r.json()
                