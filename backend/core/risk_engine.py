class RiskEngine:
    """Rule-based risk scoring."""
    
    def compute(self, driver_state: dict, road_state: dict, temporal_stats: dict) -> dict:
        ear_mean = temporal_stats.get('ear_mean', 0.0)
        perclos = temporal_stats.get('perclos', 0.0)
        mar_mean = temporal_stats.get('mar_mean', 0.0)
        pitch_mean = temporal_stats.get('pitch_mean', 0.0)
        yaw_mean = temporal_stats.get('yaw_mean', 0.0)
        ear_trend = temporal_stats.get('ear_trend', 0.0)
        
        # Fatigue score
        f_part1 = 40 * perclos
        f_part2 = 30 * (1 - ear_mean) if ear_mean < 0.4 else 0
        f_part3 = 20 if mar_mean > 0.5 else 0
        f_part4 = 10 if abs(pitch_mean) > 15 else 0
        fatigue_score = f_part1 + f_part2 + f_part3 + f_part4
        fatigue_score = min(100, max(0, fatigue_score))
        
        # Attention score
        attention_score = 0
        if abs(yaw_mean) > 15:
            attention_score += 30
        if abs(pitch_mean) > 10:
            attention_score += 20
        if ear_trend < 0:
            attention_score += 20
        attention_score = min(100, max(0, attention_score))
        
        # Hazard score
        hazard_score = road_state.get('hazard_score', 0.0)
        
        # Proximity score
        nearest = road_state.get('nearest_distance_m', 100.0)
        if nearest < 3:
            proximity_score = 100
        elif nearest < 5:
            proximity_score = 80
        elif nearest < 8:
            proximity_score = 60
        elif nearest < 15:
            proximity_score = 40
        elif nearest < 30:
            proximity_score = 20
        else:
            proximity_score = 0
            
        final_risk = int(0.35 * fatigue_score + 0.20 * attention_score + 0.25 * hazard_score + 0.20 * proximity_score)
        final_risk = min(100, max(0, final_risk))
        
        if final_risk <= 30:
            status = "SAFE"
            alert = "✅ All systems normal — drive safe"
        elif final_risk <= 55:
            status = "CAUTION"
            alert = "⚠️ CAUTION — Stay alert, conditions changing"
        elif final_risk <= 75:
            status = "HIGH"
            alert = "⚠️ HIGH RISK — Reduce speed and take a break"
        else:
            status = "CRITICAL"
            alert = "🚨 CRITICAL — REDUCE SPEED IMMEDIATELY"
            
        return {
            "risk": final_risk,
            "status": status,
            "alert": alert,
            "components": {
                "fatigue_score": fatigue_score,
                "attention_score": attention_score,
                "hazard_score": hazard_score,
                "proximity_score": proximity_score
            }
        }
