import time
from src.event_bus import EventBus, Event

class BaseAgent:
    """Base definition for platform agents communicating via the event bus."""
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus

    async def handle_event(self, event: Event):
        raise NotImplementedError("Agents must implement handle_event")

    async def send_event(self, event_type: str, recipient: str, payload: dict):
        event = Event(
            event_id=f"{self.name}-{int(time.time() * 1000)}",
            event_type=event_type,
            sender=self.name,
            recipient=recipient,
            payload=payload
        )
        await self.event_bus.publish(event)
