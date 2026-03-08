import time
from enum import Enum
from loguru import logger

class RoomState(Enum):
    UNKNOWN = "unknown"
    OCCUPIED = "occupied"       # person detected
    EMPTY = "empty"             # person not detected
    JUST_LEFT = "just_left"     # transition: was occupied, now empty
    JUST_ENTERED = "just_entered"  # transition: was empty, now occupied

class RoomStateTracker:
    """
    Tracks room occupancy over time using YOLO person detections.
    Handles the "no detection for a few frames ≠ person left" problem
    with a grace period before declaring the room empty.
    """
    
    def __init__(self, 
                 empty_grace_seconds: float = 3.0,
                 person_class: str = "person"):
        self.person_class = person_class
        self.empty_grace_seconds = empty_grace_seconds
        
        self.current_state = RoomState.UNKNOWN
        self.last_person_seen_at: float = None
        self.state_entered_at: float = time.time()
        self.absence_start_at: float = None
        
    
    def _person_in_detections(self, detections: list) -> bool:
        return any(
            d["class_name"] == self.person_class and d["confidence"] > 0.5
            for d in detections
        )
    
    def update(self, detections: list) -> tuple[RoomState, dict | None]:
        """
        Feed in current frame's detections.
        Returns (new_state, event_dict | None)
        event_dict is non-None only on state transitions.
        """
        now = time.time()
        person_present = self._person_in_detections(detections)
        event = None
        
        if person_present:
            self.last_person_seen_at = now
            
            if self.current_state in (RoomState.EMPTY, 
                                       RoomState.JUST_LEFT, 
                                       RoomState.UNKNOWN):
                # TRANSITION: room was empty, now occupied
                duration_away = None
                if self.absence_start_at:
                    duration_away = now - self.absence_start_at
                
                self.current_state = RoomState.JUST_ENTERED
                self.state_entered_at = now
                
                event = {
                    "type": "person_entered",
                    "timestamp": now,
                    "duration_away_seconds": duration_away,
                }
                logger.info(
                    f"ENTERED room" + 
                    (f" after {duration_away:.0f}s away" if duration_away else "")
                )
            else:
                # Already occupied — stay occupied
                self.current_state = RoomState.OCCUPIED
        
        else:
            # No person detected this frame
            if self.current_state in (RoomState.OCCUPIED, RoomState.JUST_ENTERED):
                # Start the grace period countdown
                time_since_last_seen = (
                    now - self.last_person_seen_at 
                    if self.last_person_seen_at 
                    else float('inf')
                )
                
                if time_since_last_seen > self.empty_grace_seconds:
                    # Grace period expired — person actually left
                    self.current_state = RoomState.JUST_LEFT
                    self.absence_start_at = self.last_person_seen_at

                    duration_in_room = self.last_person_seen_at - self.state_entered_at
                    
                    event = {
                        "type": "person_left",
                        "timestamp": now,
                        "was_in_room_since": self.state_entered_at,
                        "duration_in_room_seconds": duration_in_room,
                    }
                    logger.info(
                        f"LEFT room after "
                        f"{duration_in_room:.0f}s inside"
                    )
            
            elif self.current_state == RoomState.JUST_LEFT:
                self.current_state = RoomState.EMPTY
    
        return self.current_state, event