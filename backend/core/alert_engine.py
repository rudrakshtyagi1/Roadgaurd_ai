import collections
import time
import uuid
from datetime import datetime, timezone

class AlertEngine:
    """Alert and event logging."""
    
    def __init__(self):
        self._events = collections.deque(maxlen=100)
        self._last_alerts = {}
        self.DEBOUNCE_SECONDS = 3.0
        
    def generate_alert(self, risk_data: dict) -> dict:
        status = risk_data.get('status', 'SAFE')
        if status in ['SAFE', 'CAUTION']:
            return None
            
        alert_type = 'CRITICAL_RISK' if status == 'CRITICAL' else 'HIGH_RISK'
        now = time.time()
        
        if alert_type in self._last_alerts:
            if now - self._last_alerts[alert_type] < self.DEBOUNCE_SECONDS:
                return None
                
        self._last_alerts[alert_type] = now
        return self.log_event(alert_type, risk_data.get('alert', 'Warning'), status)
        
    def log_event(self, event_type: str, message: str, risk_level: str) -> dict:
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "message": message,
            "risk_level": risk_level
        }
        self._events.append(event)
        return event
        
    def get_recent_events(self, n: int = 20) -> list:
        events_list = list(self._events)
        return events_list[-n:]
        
    def clear_events(self):
        self._events.clear()
