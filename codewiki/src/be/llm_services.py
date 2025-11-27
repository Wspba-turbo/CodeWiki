"""
LLM service factory for creating configured LLM clients.
Supports OpenAI-compatible APIs, native Anthropic API, and Google Gemini API.
"""
from typing import Union, Optional

from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types as genai_types

from codewiki.src.config import Config


def is_anthropic_provider(config: Config) -> bool:
    """Check if the configuration specifies Anthropic as the provider."""
    return config.provider.lower() == "anthropic"


def is_google_provider(config: Config) -> bool:
    """Check if the configuration specifies Google as the provider."""
    return config.provider.lower() == "google"


def _create_anthropic_provider(api_key: str, base_url: Optional[str] = None) -> AnthropicProvider:
    """Create AnthropicProvider with async client and custom timeout."""
    import httpx
    async_client = AsyncAnthropic(
        api_key=api_key,
        base_url=base_url if base_url else None,
        timeout=httpx.Timeout(600.0, connect=30.0)  # 10 minutes timeout
    )
    return AnthropicProvider(anthropic_client=async_client)


def _create_google_provider(api_key: str) -> GoogleProvider:
    """Create GoogleProvider with API key."""
    return GoogleProvider(api_key=api_key)


def create_main_model(config: Config) -> Union[OpenAIModel, AnthropicModel, GoogleModel]:
    """Create the main LLM model from configuration."""
    if is_anthropic_provider(config):
        return AnthropicModel(
            model_name=config.main_model,
            provider=_create_anthropic_provider(config.llm_api_key, config.llm_base_url),
            settings=AnthropicModelSettings(
                temperature=0.0,
                max_tokens=32768
            )
        )
    elif is_google_provider(config):
        return GoogleModel(
            model_name=config.main_model,
            provider=_create_google_provider(config.llm_api_key),
            settings=GoogleModelSettings(
                temperature=0.0,
                max_output_tokens=32768
            )
        )
    else:
        return OpenAIModel(
            model_name=config.main_model,
            provider=OpenAIProvider(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key
            ),
            settings=OpenAIModelSettings(
                temperature=0.0,
                max_tokens=32768
            )
        )


def create_fallback_model(config: Config) -> Union[OpenAIModel, AnthropicModel, GoogleModel]:
    """Create the fallback LLM model from configuration."""
    if is_anthropic_provider(config):
        return AnthropicModel(
            model_name=config.fallback_model,
            provider=_create_anthropic_provider(config.llm_api_key, config.llm_base_url),
            settings=AnthropicModelSettings(
                temperature=0.0,
                max_tokens=32768
            )
        )
    elif is_google_provider(config):
        return GoogleModel(
            model_name=config.fallback_model,
            provider=_create_google_provider(config.llm_api_key),
            settings=GoogleModelSettings(
                temperature=0.0,
                max_output_tokens=32768
            )
        )
    else:
        return OpenAIModel(
            model_name=config.fallback_model,
            provider=OpenAIProvider(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key
            ),
            settings=OpenAIModelSettings(
                temperature=0.0,
                max_tokens=32768
            )
        )


def create_fallback_models(config: Config) -> FallbackModel:
    """Create fallback models chain from configuration."""
    main = create_main_model(config)
    fallback = create_fallback_model(config)
    return FallbackModel(main, fallback)


def create_openai_client(config: Config) -> OpenAI:
    """Create OpenAI client from configuration."""
    return OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key
    )


def create_anthropic_client(config: Config) -> Anthropic:
    """Create Anthropic client from configuration."""
    import httpx
    return Anthropic(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url if config.llm_base_url else None,
        timeout=httpx.Timeout(600.0, connect=30.0)  # 10 minutes timeout
    )


def create_google_client(config: Config) -> genai.Client:
    """Create Google GenAI client from configuration."""
    return genai.Client(api_key=config.llm_api_key)


def call_llm(
    prompt: str,
    config: Config,
    model: str = None,
    temperature: float = 0.0
) -> str:
    """
    Call LLM with the given prompt.
    Uses the provider specified in config (OpenAI-compatible, native Anthropic, or Google Gemini).

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting

    Returns:
        LLM response text
    """
    if model is None:
        model = config.main_model

    if is_anthropic_provider(config):
        # Use Anthropic client
        client = create_anthropic_client(config)
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=32768
        )
        return response.content[0].text
    elif is_google_provider(config):
        # Use Google GenAI client (new SDK: from google import genai)
        client = create_google_client(config)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=32768
            )
        )
        return response.text
    else:
        # Use OpenAI client
        client = create_openai_client(config)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=32768
        )
        return response.choices[0].message.content
