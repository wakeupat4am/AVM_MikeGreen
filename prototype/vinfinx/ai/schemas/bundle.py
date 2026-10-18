"""EvidenceBundle: the signed, time-bounded input to the decision engine (decision D8: HMAC-SHA256)."""
from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import StrictModel, canonical_json, content_hash
from .evidence import EvidenceItem
from .policy import PolicyResult

KEY_ENV = "VINFINX_BUNDLE_KEY"  # hex-encoded secret, at least 32 bytes
KEY_ID_ENV = "VINFINX_BUNDLE_KEY_ID"


class BundleSignature(StrictModel):
    alg: Literal["HMAC-SHA256"] = "HMAC-SHA256"
    key_id: str
    value: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceBundle(StrictModel):
    schema_version: Literal["evidence_bundle/0.1"] = "evidence_bundle/0.1"
    bundle_id: str
    case_id: str
    created_at: AwareDatetime
    expires_at: AwareDatetime
    items: list[EvidenceItem] = Field(min_length=1)
    policy: PolicyResult
    signature: BundleSignature | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvidenceBundle:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        ids = [i.evidence_id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate evidence_id in bundle")
        if self.policy.case_id != self.case_id:
            raise ValueError("policy result belongs to a different case")
        missing = self.policy.cited_evidence_ids - set(ids)
        if missing:
            raise ValueError(f"policy cites evidence not in the bundle: {sorted(missing)}")
        return self

    @property
    def mapping_id(self) -> str:
        return self.policy.mapping_id

    @property
    def evidence_ids(self) -> list[str]:
        return [i.evidence_id for i in self.items]

    def signing_payload(self) -> bytes:
        return canonical_json(self.model_dump(mode="json", exclude={"signature"}))

    @property
    def bundle_hash(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"signature"}))

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


def sign_bundle(bundle: EvidenceBundle, key: bytes, key_id: str) -> EvidenceBundle:
    if len(key) < 32:
        raise ValueError("signing key must be at least 32 bytes")
    mac = hmac.new(key, bundle.signing_payload(), hashlib.sha256).hexdigest()
    return bundle.model_copy(update={"signature": BundleSignature(key_id=key_id, value=mac)})


def verify_bundle(bundle: EvidenceBundle, keys: dict[str, bytes]) -> bool:
    """True only if the bundle is signed with a known key and unchanged since signing."""
    if bundle.signature is None or bundle.signature.key_id not in keys:
        return False
    expected = hmac.new(keys[bundle.signature.key_id], bundle.signing_payload(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, bundle.signature.value)


def key_from_env() -> tuple[bytes, str]:
    """Signing key and key ID from the environment. Keys are never stored in the repository."""
    try:
        key, key_id = bytes.fromhex(os.environ[KEY_ENV]), os.environ[KEY_ID_ENV]
    except KeyError as missing:
        raise RuntimeError(f"set {missing} to sign or verify evidence bundles") from None
    if len(key) < 32:
        raise RuntimeError(f"{KEY_ENV} must be at least 32 bytes (64 hex characters)")
    return key, key_id
