import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .base import ApiKeyMixin, BaseSearchEngine, SearchResult, mask_api_key

logger = logging.getLogger(__name__)

BOCHA_ERROR_LOG_BODY_MAX_LENGTH = 512


def _truncate_for_log(text: str) -> str:
    if len(text) <= BOCHA_ERROR_LOG_BODY_MAX_LENGTH:
        return text
    return f"{text[:BOCHA_ERROR_LOG_BODY_MAX_LENGTH]}... [truncated]"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _merge_texts(*values: str) -> str:
    """按顺序合并非空文本,避免把 snippet 和 summary 重复塞给总结 prompt。"""
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if text and text not in seen:
            merged.append(text)
            seen.add(text)
    return "\n".join(merged)


class BochaEngine(BaseSearchEngine, ApiKeyMixin):
    """Bocha Web Search API client."""

    BASE_URL = "https://api.bochaai.com"
    SEARCH_ENDPOINT = "/v1/web-search"
    API_MAX_RESULTS = 50

    freshness: Optional[str]
    summary: bool

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._init_api_keys(self.config, "BOCHA_API_KEY")

        freshness_cfg = self.config.get("freshness")
        if isinstance(freshness_cfg, str):
            freshness_cfg = freshness_cfg.strip()
        self.freshness = freshness_cfg or None
        self.summary = bool(self.config.get("summary", True))

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        """Execute a Bocha web-search request."""
        api_keys = self._iter_api_keys()
        if not api_keys:
            logger.warning("Bocha API key is not configured; skip Bocha search.")
            return []

        request_count = self._request_count(num_results)
        payload: Dict[str, Any] = {
            "query": query,
            "summary": self.summary,
            "count": request_count,
        }
        if self.freshness:
            payload["freshness"] = self.freshness

        timeout = aiohttp.ClientTimeout(total=self.TIMEOUT)
        headers_base = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for api_key in api_keys:
                headers = {
                    **headers_base,
                    "Authorization": f"Bearer {api_key}",
                }
                try:
                    async with session.post(
                        f"{self.BASE_URL}{self.SEARCH_ENDPOINT}",
                        json=payload,
                        headers=headers,
                        proxy=self.proxy,
                    ) as response:
                        response_text = await response.text()
                        if response.status >= 400:
                            logger.error(
                                "Bocha search request failed with status %s for key %s; response body: %s",
                                response.status,
                                mask_api_key(api_key),
                                _truncate_for_log(response_text),
                            )
                            continue

                        if not response_text:
                            logger.error("Bocha returned an empty response for key %s.", mask_api_key(api_key))
                            continue

                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError:
                            logger.error(
                                "Failed to parse Bocha response as JSON for key %s: %s",
                                mask_api_key(api_key),
                                _truncate_for_log(response_text),
                            )
                            continue

                except Exception as exc:
                    logger.error(
                        "Bocha search raised an exception for key %s: %s",
                        mask_api_key(api_key),
                        exc,
                        exc_info=True,
                    )
                    continue

                if not isinstance(data, dict):
                    logger.error(
                        "Unexpected Bocha response type for key %s: %s",
                        mask_api_key(api_key),
                        type(data),
                    )
                    continue

                web_pages = data.get("webPages")
                if web_pages is None:
                    return []
                if not isinstance(web_pages, dict):
                    logger.error(
                        "Unexpected Bocha webPages type for key %s: %s",
                        mask_api_key(api_key),
                        type(web_pages),
                    )
                    continue

                items = web_pages.get("value")
                if items is None:
                    return []
                if not isinstance(items, list):
                    logger.error(
                        "Unexpected Bocha webPages.value type for key %s: %s",
                        mask_api_key(api_key),
                        type(items),
                    )
                    continue

                results = self._parse_results(items)
                return results[:request_count]

        return []

    def _request_count(self, num_results: int) -> int:
        configured_max = self.max_results if self.max_results > 0 else self.API_MAX_RESULTS
        requested = num_results if num_results > 0 else configured_max
        return max(1, min(requested, configured_max, self.API_MAX_RESULTS))

    def _parse_results(self, items: List[Any]) -> List[SearchResult]:
        results: List[SearchResult] = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            title = self.tidy_text(_to_text(item.get("name")))
            url = _to_text(item.get("url"))
            if not title or not self._is_valid_url(url):
                continue

            snippet = self.tidy_text(_to_text(item.get("snippet")))
            summary = self.tidy_text(_to_text(item.get("summary")))
            abstract = _merge_texts(snippet, summary)

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    abstract=abstract,
                    rank=index,
                    content=summary,
                )
            )

        return results
