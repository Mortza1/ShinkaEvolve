import backoff
import openai
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"Ollama - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_ollama(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Ollama model using standard OpenAI chat completions API."""
    messages = [
        {"role": "system", "content": system_msg},
        *msg_history,
        {"role": "user", "content": msg}
    ]

    # Remove unsupported parameters for Ollama
    ollama_kwargs = {}
    if "temperature" in kwargs:
        ollama_kwargs["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        ollama_kwargs["max_tokens"] = kwargs["max_tokens"]
    elif "max_output_tokens" in kwargs:
        ollama_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **ollama_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content}
        ]
    else:
        # Structured output for Ollama - use JSON mode
        ollama_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **ollama_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content}
        ]

    # Local models are free
    cost = 0.0
    input_cost = 0.0
    output_cost = 0.0

    input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
    output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result
