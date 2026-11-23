"""Sync API authentication — one credential bound to one LOCATION_CODE."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass

from django.conf import settings
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from city_sync.models import SyncCredential

logger = logging.getLogger(__name__)
_ENV_KEY_WARNED: set[str] = set()


@dataclass
class SyncPrincipal:
    location_code: str
    is_authenticated: bool = True

    @property
    def pk(self):
        return self.location_code


class SyncApiKeyAuthentication(BaseAuthentication):
    """
    Authorization: Bearer <raw-key>
    or X-CIIS-Sync-Key: <raw-key>
    """

    header_names = ("HTTP_AUTHORIZATION", "HTTP_X_CIIS_SYNC_KEY")

    def authenticate(self, request):
        raw = self._extract_key(request)
        if not raw:
            return None

        # Dev fallback: SYNC_API_KEYS="DI_KHAN=secret,ABBOTTABAD=secret"
        env_map = self._env_key_map()
        key_hash = SyncCredential.hash_key(raw)

        cred = (
            SyncCredential.objects.filter(key_hash=key_hash, is_active=True, revoked_at__isnull=True)
            .first()
        )
        if cred:
            return (SyncPrincipal(location_code=cred.location_code), raw)

        for loc, secret in env_map.items():
            if secret and hmac.compare_digest(secret.encode(), raw.encode()):
                if not self._env_key_allowed(loc):
                    # The site has a key on the Locations page (active, rotated or revoked):
                    # the old .env key must not bypass a rotation or a removal.
                    logger.warning("Refused SYNC_API_KEYS .env key for %s: it has a managed key", loc)
                    break
                return (SyncPrincipal(location_code=loc), raw)

        raise AuthenticationFailed("Invalid sync API key")

    @staticmethod
    def _env_key_allowed(loc: str) -> bool:
        """The .env fallback works in DEBUG, or for a city that has no SyncCredential row at all."""
        if getattr(settings, "DEBUG", False):
            return True
        if SyncCredential.objects.filter(location_code=loc).exists():
            return False
        if loc not in _ENV_KEY_WARNED:
            _ENV_KEY_WARNED.add(loc)
            logger.warning(
                "City %s authenticates with the SYNC_API_KEYS .env fallback; add it on the "
                "Locations page so its key can be rotated and revoked.",
                loc,
            )
        return True

    def _extract_key(self, request) -> str:
        auth = request.META.get("HTTP_AUTHORIZATION", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (request.META.get("HTTP_X_CIIS_SYNC_KEY") or "").strip()

    def _env_key_map(self) -> dict[str, str]:
        raw = getattr(settings, "SYNC_API_KEYS", "") or ""
        out: dict[str, str] = {}
        for part in str(raw).split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            loc, secret = part.split("=", 1)
            out[loc.strip().upper()] = secret.strip()
        return out
