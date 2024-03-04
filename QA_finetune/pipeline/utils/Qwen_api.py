from http import HTTPStatus
import dashscope
import time
from datetime import datetime
dashscope.api_key="sk-d83c8efeee9449198ef0a4f5c67be150"

def make_requests(
        engine,
        prompts,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop_sequences,
        logprobs,
        n,
        best_of,
        max_tokens,
    ):
    responses = []
    for prompt in prompts:
        response = dashscope.Generation.call(
            model=engine,
            prompt=prompt,
            result_format='message',
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of,
            seed=2024228,
            max_tokens=max_tokens,
            )

        if response.status_code == HTTPStatus.OK:
            responses.append(response)
        else:
            print("failed")
    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": responses[j]["output"]["choices"][0]["message"]["content"] } if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]

    
if __name__=="__main__":
    make_requests(["1+1=","hello world"],0,0.1,0,0,["stop"],1,1,1)