"""
API utilities for managing and interacting with various AI provider APIs.

This module provides a robust framework for handling multiple API keys,
rate limiting, automatic retries, and provider fallbacks. It is designed
to make API interactions resilient and efficient in a production environment.

Key Components:
- APIKeyManager: Manages a pool of API keys, prioritizing free-tier keys
  and handling rate-limit feedback.
- RateLimiter: Enforces rate limits per key for requests per minute/second
  and maximum concurrent requests.
- APIClientFactory: Creates and caches API clients on demand.
- execute_with_retry: A high-level wrapper that executes an API operation
  with automatic retries, key rotation, and provider fallbacks.
"""
import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncGenerator, Deque, Tuple

import redis

logger = logging.getLogger(__name__)


@dataclass
class APIKeyConfig:
    """Configuration for a single API key."""
    key: str
    is_free_tier: bool = True
    # Default rate limits are conservative; they can be tuned.
    rate_limit_requests_per_minute: int = 50
    rate_limit_requests_per_second: int = 2
    max_concurrent: int = 4


@dataclass
class APIKeyStatus:
    """Tracks the operational status of an API key."""
    config: APIKeyConfig
    retry_after_timestamp: float = 0.0
    last_error: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_available(self) -> bool:
        """Checks if the key is currently available for use."""
        return time.time() >= self.retry_after_timestamp


class APIKeyManager:
    """
    Manages a pool of API keys with rate-limit tracking and prioritization.

    This class handles the rotation of multiple API keys, prioritizing free-tier
    keys over premium ones to manage costs. It deactivates keys that hit a
    rate limit for a specified duration.
    Implements premium-only fallback for 1 hour when all free keys are exhausted.
    """

    def __init__(self, provider: str, api_keys_str: str, premium_key: Optional[str] = None):
        self.provider = provider.lower()
        self._keys: Dict[str, APIKeyStatus] = {}
        self._key_queue: Deque[str] = deque()
        self._lock = asyncio.Lock()

        # Always trim all keys and premium key
        all_keys = {k.strip() for k in api_keys_str.split(',') if k.strip()}
        premium_key = premium_key.strip() if premium_key else None
        if premium_key and premium_key in all_keys:
            all_keys.remove(premium_key)
        if premium_key:
            all_keys.add(premium_key)

        if not all_keys:
            raise ValueError(f"No API keys provided for provider: {self.provider}")

        self.premium_key = premium_key
        self._free_keys = [k for k in all_keys if k != premium_key]
        self._paid_keys = [k for k in all_keys if k == premium_key]

        # Add free keys first so they are prioritized in the queue
        for key in self._free_keys:
            config = APIKeyConfig(key=key, is_free_tier=True)
            self._keys[key] = APIKeyStatus(config)

        for key in self._paid_keys:
            config = APIKeyConfig(
                key=key,
                is_free_tier=False,
                rate_limit_requests_per_minute=100,  # Higher limits for paid keys
                rate_limit_requests_per_second=5,
                max_concurrent=10
            )
            self._keys[key] = APIKeyStatus(config)

        # The queue determines the order of key selection
        self._key_queue.extend(self._free_keys)
        self._key_queue.extend(self._paid_keys)

        # Premium fallback state
        self._premium_only_mode = False
        self._premium_only_until = 0.0

        logger.info(
            f"Initialized {self.provider} APIKeyManager with {len(self._keys)} keys "
            f"({len(self._free_keys)} free, {len(self._paid_keys)} paid)."
        )

    async def get_next_available_key(self) -> Optional[APIKeyStatus]:
        """
        Retrieves the next available API key status object from the pool.
        If all free keys are exhausted, switches to premium key for 1 hour (persisted in Redis).
        During this time, only premium key is used. After 1 hour, free keys are retried.
        """
        async with self._lock:
            now = time.time()
            redis_client = get_premium_redis()
            redis_key = f"premium_only_mode:{self.provider}"
            premium_until = 0.0
            if redis_client:
                try:
                    val = redis_client.get(redis_key)
                    if val:
                        premium_until = float(val)
                except Exception as e:
                    logger.warning(f"Redis premium mode read failed: {e}")
            # Check if premium-only mode should be reset
            if (self._premium_only_mode or (premium_until and now < premium_until)):
                if premium_until and now >= premium_until:
                    self._premium_only_mode = False
                    self._premium_only_until = 0.0
                    if redis_client:
                        try:
                            redis_client.delete(redis_key)
                        except Exception as e:
                            logger.warning(f"Redis premium mode delete failed: {e}")
                    logger.info(f"Premium-only mode ended for provider '{self.provider}'. Free keys will be retried.")
                else:
                    self._premium_only_mode = True
                    self._premium_only_until = premium_until
            # If in premium-only mode, only return premium key if available
            if self._premium_only_mode:
                if self.premium_key and self.premium_key in self._keys:
                    key_status = self._keys[self.premium_key]
                    if key_status.is_available:
                        return key_status
                    else:
                        # Mark premium key as unavailable for a short period if rate-limited
                        key_status.retry_after_timestamp = now + 60
                return None
            # Try free keys first
            free_available = []
            for key in self._free_keys:
                if key in self._keys and self._keys[key].is_available:
                    free_available.append(self._keys[key])
            if free_available:
                # Rotate to balance usage
                self._key_queue.rotate(-1)
                for _ in range(len(self._key_queue)):
                    key_str = self._key_queue[0]
                    if key_str in self._free_keys and self._keys[key_str].is_available:
                        return self._keys[key_str]
                    self._key_queue.rotate(-1)
                return free_available[0]
            # No free keys available, activate premium-only mode
            if self.premium_key and self.premium_key in self._keys:
                self._premium_only_mode = True
                self._premium_only_until = now + 3600  # 1 hour
                if redis_client:
                    try:
                        redis_client.set(redis_key, str(self._premium_only_until), ex=3700)
                    except Exception as e:
                        logger.warning(f"Redis premium mode set failed: {e}")
                logger.warning(
                    f"All free API keys exhausted for provider '{self.provider}'. Switching to premium key for 1 hour.")
                key_status = self._keys[self.premium_key]
                if key_status.is_available:
                    return key_status
                else:
                    # Mark premium key as unavailable for a short period if rate-limited
                    key_status.retry_after_timestamp = now + 60
            return None

    def mark_rate_limited(self, key: str, retry_after_seconds: int, error_msg: Optional[str] = None):
        """
        Marks an API key as rate-limited.

        Args:
            key: The API key string to mark.
            retry_after_seconds: The duration for which the key should be considered unavailable.
            error_msg: The error message associated with the rate limit.
        """
        if key in self._keys:
            status = self._keys[key]
            status.retry_after_timestamp = time.time() + retry_after_seconds
            status.last_error = error_msg
            logger.warning(
                f"Provider '{self.provider}' key ending in '...{key[-4:]}' is rate-limited. "
                f"Retrying after {retry_after_seconds}s. Error: {error_msg}"
            )


class RateLimiter:
    """
    Client-side rate limiter to avoid hitting API limits.
    """

    def __init__(self, config: APIKeyConfig):
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._requests_per_second_lock = asyncio.Lock()
        self._last_request_time = 0

    async def __aenter__(self):
        await self._semaphore.acquire()
        try:
            async with self._requests_per_second_lock:
                current_time = time.time()
                time_since_last = current_time - self._last_request_time
                required_interval = 1.0 / self._config.rate_limit_requests_per_second
                if time_since_last < required_interval:
                    await asyncio.sleep(required_interval - time_since_last)
                self._last_request_time = time.time()
        except Exception:
            self._semaphore.release()  # Release semaphore if something goes wrong
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


class APIClientFactory:
    """
    Factory for creating and managing API clients with integrated key rotation and rate limiting.
    """

    def __init__(self, provider: str, api_keys_str: str, premium_key: Optional[str] = None):
        self.provider = provider.lower()
        self.key_manager = APIKeyManager(self.provider, api_keys_str, premium_key)
        self._clients: Dict[str, Any] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_client_session(self) -> Optional[Tuple[Any, APIKeyStatus, RateLimiter]]:
        """
        Gets an API client, its status object, and its rate limiter.

        Returns:
            A tuple containing the client, its status, and its rate limiter, or None if no keys are available.
        """
        key_status = await self.key_manager.get_next_available_key()
        if not key_status:
            return None

        key = key_status.config.key
        async with self._lock:
            if key not in self._clients:
                try:
                    self._rate_limiters[key] = RateLimiter(key_status.config)
                    # Lazily import and initialize clients
                    if self.provider == "gemini":
                        import google.generativeai as genai
                        genai.configure(api_key=key)
                        model_name = os.getenv("GEMINI_MODEL_VERSION", "gemini-1.5-flash")
                        self._clients[key] = genai.GenerativeModel(model_name)
                    elif self.provider == "openai":
                        from openai import AsyncOpenAI
                        self._clients[key] = AsyncOpenAI(api_key=key)
                    elif self.provider == "groq":
                        from groq import AsyncGroq
                        self._clients[key] = AsyncGroq(api_key=key)
                    elif self.provider == "claude":
                        from anthropic import AsyncAnthropic
                        self._clients[key] = AsyncAnthropic(api_key=key)
                    else:
                        raise ValueError(f"Unsupported provider: {self.provider}")
                    logger.info(f"Initialized client for provider '{self.provider}' with key ...{key[-4:]}")
                except Exception as e:
                    logger.error(f"Failed to initialize client for provider '{self.provider}': {e}")
                    return None

            return self._clients[key], key_status, self._rate_limiters[key]


# Redis client for premium-only mode
_redis_premium = None


def get_premium_redis():
    global _redis_premium
    if _redis_premium is not None:
        return _redis_premium
    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")
    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    if redis_host and redis_port:
        try:
            _redis_premium = redis.Redis(
                host=redis_host,
                port=int(redis_port),
                decode_responses=True,
                username=redis_username,
                password=redis_password,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=20
            )
            _redis_premium.ping()
            return _redis_premium
        except Exception as e:
            logger.warning(f"Redis unavailable for premium-only mode: {e}")
    return None


async def execute_with_retry(
        operation: callable,
        client_factories: List[APIClientFactory],
        max_retries_per_provider: int = 3,
        initial_backoff_secs: float = 1.0,
) -> AsyncGenerator[Any, None]:
    """
    Executes an API operation with a robust retry and NO fallback strategy.
    Only the primary provider is used. If all keys are exhausted, raises error.
    """
    last_exception = None
    if not client_factories:
        raise IOError("No API client factories configured.")
    factory = client_factories[0]  # Only use the primary provider
    for attempt in range(max_retries_per_provider):
        try:
            session = await factory.get_client_session()
            if not session:
                logger.warning(f"No available API keys for provider: {factory.provider}.")
                break
            client, key_status, rate_limiter = session
            async with key_status.lock:
                async with rate_limiter:
                    async for result in operation(client):
                        yield result
                    return
        except Exception as e:
            last_exception = e
            retry_after = 60
            error_str = str(e).lower()
            if (hasattr(e, 'status_code') and e.status_code == 429) or \
                    'rate limit' in error_str or 'quota' in error_str:
                if hasattr(e, 'retry_after'):
                    retry_after = int(e.retry_after) + 1
                if session:
                    factory.key_manager.mark_rate_limited(session[1].config.key, retry_after, str(e))
            sleep_time = initial_backoff_secs * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries_per_provider} for provider '{factory.provider}' failed. "
                f"Retrying in {sleep_time:.2f}s. Error: {e}"
            )
            await asyncio.sleep(sleep_time)
            continue
    logger.critical(f"All attempts failed for provider: {factory.provider}. Last error: {last_exception}")
    if last_exception:
        raise last_exception
    else:
        raise IOError("All API keys for the primary provider are unavailable.")
