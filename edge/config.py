"""Validated environment settings with local-only web defaults."""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class EdgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGE_CAMERA_", extra="ignore", hide_input_in_errors=True
    )

    source: Literal["picamera", "rtsp", "synthetic"] = "picamera"
    rtsp_url: str | None = None
    frame_width: int = Field(default=1280, gt=0, le=4096)
    frame_height: int = Field(default=720, gt=0, le=4096)
    capture_fps: PositiveFloat = 10.0
    analysis_fps: PositiveFloat = 3.0

    embedder: Literal["insightface", "fake"] = "insightface"
    insightface_model: str = Field(default="buffalo_s", pattern=r"^[A-Za-z0-9_-]+$")
    insightface_root: Path = Path("models/insightface")
    match_max_distance: float = Field(default=0.55, ge=0, le=2, allow_inf_nan=False)
    consecutive_frames: int = Field(default=2, ge=1)

    motion_enabled: bool = True
    motion_min_change: UnitFloat = 0.01
    motion_cooldown_s: NonnegativeFloat = 10.0

    detector: Literal["none", "hailo"] = "none"
    hef_path: str = "models/person-cat.hef"
    detect_confidence: UnitFloat = 0.5
    notify_on_person: bool = True
    notify_on_cat: bool = False

    clip_enabled: bool = True
    clip_pre_seconds: NonnegativeFloat = 30.0
    clip_post_seconds: NonnegativeFloat = 30.0
    clip_max_seconds: PositiveFloat = 300.0
    clip_on_known: bool = True

    data_dir: Path = Path("data")
    # Email is opt-in and sends event snapshots outside this machine.
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = ""
    smtp_password: str = Field(default="", repr=False)
    mail_from: str = ""
    mail_to: str = ""
    notify_on_known: bool = True
    notify_on_owner: bool = True
    notify_cooldown_s: NonnegativeFloat = 600.0

    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8000, ge=1, le=65535)
    web_pin: str = Field(default="", max_length=128, repr=False)
    web_cookie_secure: bool = False
    # Applies to event records/media; enrolled templates have a separate lifecycle.
    # 0 explicitly disables event retention. No encryption at rest is implemented.
    retention_days: int = Field(default=30, ge=0)

    # Optional Linux ping probe. Reachability is not proof of a person's identity.
    presence_host: str = ""
    presence_interval_s: PositiveFloat = 30.0
    presence_absent_after_s: PositiveFloat = 600.0

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        try:
            loopback = ip_address(self.web_host).is_loopback
        except ValueError:
            loopback = self.web_host.lower() == "localhost"
        if not loopback and len(self.web_pin) < 12:
            raise ValueError(
                "A passphrase of at least 12 characters is required for network binding"
            )
        if self.source == "rtsp" and not self.rtsp_url:
            raise ValueError("RTSP source requires EDGE_CAMERA_RTSP_URL")
        if self.clip_max_seconds < self.clip_post_seconds:
            raise ValueError("clip_max_seconds must be at least clip_post_seconds")
        if self.smtp_host and not (self.mail_to and (self.mail_from or self.smtp_user)):
            raise ValueError("SMTP requires a recipient and sender address")
        if self.presence_host:
            try:
                ip_address(self.presence_host)
            except ValueError as exc:
                raise ValueError(
                    "presence_host must be an IP address, not a command option"
                ) from exc
        return self
