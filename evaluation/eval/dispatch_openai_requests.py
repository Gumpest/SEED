'''
This file is copied and modified from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a.
Thanks to Graham Neubig for sharing the original code.
'''

import openai
import asyncio
from typing import Any, List, Dict

def dispatch_openai_chat_requesets(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI chat completion API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI chat completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI ChatCompletion API. See https://platform.openai.com/docs/api-reference/chat for details.
    Returns:
        List of responses from OpenAI API.
    """
    responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=completion_kwargs["temperature"],
            max_tokens=completion_kwargs["max_tokens"]
        ).choices[0].message.content
        for x in messages_list
    ]

    return responses


async def dispatch_openai_prompt_requesets(
    prompt_list: List[str],
    model: str,
    **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI text completion API asynchronously.
    
    Args:
        prompt_list: List of prompts to be sent to OpenAI text completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI text completion API. See https://platform.openai.com/docs/api-reference/completions for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.Completion.acreate(
            engine=model,
            prompt=x,
            **completion_kwargs,
        )
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)


if __name__ == "__main__":

    import openai
    client = openai.AzureOpenAI(
        azure_endpoint="https://search.bytedance.net/gpt/openapi/online/multimodal/crawl",
        api_version="2023-07-01-preview",
        api_key="SOQMfqF8PzLpukRPOEBN4MIber15KdN1",
    )
    
    chat_completion_responses = dispatch_openai_chat_requesets(
        messages_list=[
                [{"role": "user", "content": "Write a poem about asynchronous execution."}],
                [{"role": "user", "content": "Write a poem about asynchronous pirates."}],
            ],
            model="gpt-4o-2024-05-13",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,
        )

    for i, x in enumerate(chat_completion_responses):
        print(f"Chat completion response {i}:\n{x}\n\n")