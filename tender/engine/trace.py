# engine/trace.py
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time

@dataclass
class TraceCollector:
    events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)

    def stamp(self, name: str, **data):
        self.events.append({
            "t_ms": int((time.time() - self._t0) * 1000),
            "event": name,
            **data
        })

    def set_meta(self, **data):
        self.meta.update(data)

    def to_dict(self) -> Dict[str, Any]:
        return {"meta": self.meta, "events": self.events}
