"""The ML API must refuse requests when ML_API_KEY is missing (fail closed)."""

import unittest
from types import SimpleNamespace

from fastapi import HTTPException

import api_server


def _req(path="/live/status", method="GET", headers=None, query=None):
    return SimpleNamespace(
        url=SimpleNamespace(path=path),
        method=method,
        headers={k.lower(): v for k, v in (headers or {}).items()},
        query_params=query or {},
    )


class FailClosedTests(unittest.TestCase):
    def setUp(self):
        self._saved = (api_server._API_KEY, api_server._ALLOW_NO_KEY, api_server._OPEN_PATHS)

    def tearDown(self):
        api_server._API_KEY, api_server._ALLOW_NO_KEY, api_server._OPEN_PATHS = self._saved

    def test_no_key_configured_refuses(self):
        api_server._API_KEY, api_server._ALLOW_NO_KEY = "", False
        with self.assertRaises(HTTPException) as ctx:
            api_server.require_api_key(_req())
        self.assertEqual(ctx.exception.status_code, 503)

    def test_no_key_allowed_only_with_explicit_flag(self):
        api_server._API_KEY, api_server._ALLOW_NO_KEY = "", True
        api_server.require_api_key(_req())  # no exception

    def test_health_needs_key_by_default(self):
        api_server._API_KEY, api_server._ALLOW_NO_KEY = "k" * 20, False
        api_server._OPEN_PATHS = frozenset({"/"})
        with self.assertRaises(HTTPException):
            api_server.require_api_key(_req("/health"))
        api_server.require_api_key(_req("/health", headers={"X-ML-API-Key": "k" * 20}))
        api_server.require_api_key(_req("/"))  # root stays open

    def test_wrong_key_refused(self):
        api_server._API_KEY = "right-key-value"
        with self.assertRaises(HTTPException) as ctx:
            api_server.require_api_key(_req(headers={"X-ML-API-Key": "wrong"}))
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
