import asyncio
import json
import time
from typing import Callable, Dict, List, Awaitable
from dataclasses import dataclass, asdict

@dataclass
class Event:
    event_id: str
    event_type: str       # "TASK_PLANNED", "TASK_DISPATCHED", "TASK_COMPLETED", "TASK_REJECTED"
    sender: str           # "orchestrator", "planner", "builder", "reviewer"
    recipient: str        # "orchestrator", "planner", "builder", "reviewer", "all"
    payload: dict
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def serialize(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def deserialize(cls, data: str) -> "Event":
        obj = json.loads(data)
        return cls(**obj)


class EventBus:
    """An asynchronous, in-memory publish-subscribe event routing engine."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], Awaitable[None]]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: Event):
        if event.event_type not in self._subscribers:
            return
        
        tasks = []
        for callback in self._subscribers[event.event_type]:
            # Get subscriber's name if it has one
            handler_agent_name = getattr(getattr(callback, "__self__", None), "name", None)
            if event.recipient == "all" or event.recipient == handler_agent_name or event.recipient == "orchestrator":
                tasks.append(callback(event))
        
        if tasks:
            await asyncio.gather(*tasks)
