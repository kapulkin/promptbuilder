import json
from typing import Any

import httpx
import aiohttp


class APIError(Exception):
    """General errors raised by the llm clients"""
    code: int | None = None
    status: str | None = None
    message: str | None = None
    response: httpx.Response | aiohttp.ClientResponse | None = None

    def __init__(self, code: int | None = None, response_json: dict[str, Any] = {}, response: httpx.Response | aiohttp.ClientResponse | None = None):
        self.response = response
        self.details = response_json
        self.message = self._get_message(response_json)
        self.status = self._get_status(response_json)
        self.code = code if code else self._get_code(response_json)

        super().__init__(f"{self.code} {self.status}. {self.details}")

    def _get_status(self, response_json: dict[str, Any]) -> Any:
        return response_json.get(
            "status", response_json.get("error", {}).get("status", None)
        )

    def _get_message(self, response_json: dict[str, Any]) -> Any:
        return response_json.get(
            "message", response_json.get("error", {}).get("message", None)
        )

    def _get_code(self, response_json: dict[str, Any]) -> Any:
        return response_json.get(
            "code", response_json.get("error", {}).get("code", None)
        )

    def _to_replay_record(self) -> dict[str, Any]:
        """Returns a dictionary representation of the error for replay recording."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status": self.status,
            }
        }

    @classmethod
    def raise_for_response(cls, response: httpx.Response):
        """Raises an error with detailed error message if the response has an error status."""
        if response.status_code == 200:
            return

        if isinstance(response, httpx.Response):
            try:
                response.read()
                response_json = response.json()
            except json.decoder.JSONDecodeError:
                message = response.text
                response_json = {
                    "message": message,
                    "status": response.reason_phrase,
                }
        else:
            response_json = response.body_segments[0].get('error', {})

        status_code = response.status_code
        if 400 <= status_code < 500:
            raise ClientError(status_code, response_json, response)
        elif 500 <= status_code < 600:
            raise ServerError(status_code, response_json, response)
        else:
            raise cls(status_code, response_json, response)

    @classmethod
    async def raise_for_async_response(cls, response: httpx.Response | aiohttp.ClientResponse):
        """Raises an error with detailed error message if the response has an error status."""
        status_code = 0
        response_json = None
        if isinstance(response, httpx.Response):
            if response.status_code == 200:
                return
            try:
                await response.aread()
                response_json = response.json()
            except json.decoder.JSONDecodeError:
                message = response.text
                response_json = {
                    "message": message,
                    "status": response.reason_phrase,
                }
            status_code = response.status_code
        else:
            try:
                if isinstance(response, aiohttp.ClientResponse):
                    if response.status == 200:
                        return
                    try:
                        response_json = await response.json()
                    except aiohttp.client_exceptions.ContentTypeError:
                        message = await response.text()
                        response_json = {
                            "message": message,
                            "status": response.reason,
                        }
                    status_code = response.status
                else:
                    response_json = response.body_segments[0].get("error", {})
            except ImportError:
                response_json = response.body_segments[0].get("error", {})

        if 400 <= status_code < 500:
            raise ClientError(status_code, response_json, response)
        elif 500 <= status_code < 600:
            raise ServerError(status_code, response_json, response)
        else:
            raise cls(status_code, response_json, response)


class ClientError(APIError):
    """Client error raised by the llm clients"""
    pass


class ServerError(APIError):
    """Server error raised by the llm clients"""
    pass
