"""Interfaces for future Kakao notification delivery.

No Kakao message is sent at this stage.
"""

from abc import ABC, abstractmethod

from src.config import require_environment_variable


class MessageSender(ABC):
    @abstractmethod
    def send_message(self, message: str) -> None:
        """Deliver one notification message."""


class KakaoMessageClient(MessageSender):
    """Configuration-aware placeholder for a future Kakao adapter."""

    def send_message(self, message: str) -> None:
        if not message.strip():
            raise ValueError("message must not be empty")
        require_environment_variable("KAKAO_REST_API_KEY")
        require_environment_variable("KAKAO_ACCESS_TOKEN")
        raise NotImplementedError(
            "Kakao message network calls are not implemented yet"
        )
