import numpy as np
import cv2
from collections import deque

class LightAnalyzer:
    def __init__(self,
                 low_threshold=60,
                 high_threshold=200,
                 history_size=10,
                 fluctuation_threshold=15):
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.history_size = history_size
        self.fluctuation_threshold = fluctuation_threshold
        
        self.brightness_history = deque(maxlen=history_size)

    def analyze(self, frame, overlay=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        self.brightness_history.append(brightness)

        if brightness < self.low_threshold:
            light_level = "Too Dark"
        elif brightness > self.high_threshold:
            light_level = "Too Bright"
        else:
            light_level = "Normal"

        if len(self.brightness_history) >= self.history_size:
            fluctuation = np.std(self.brightness_history)
            if fluctuation > self.fluctuation_threshold:
                stability = "Unstable"
            else:
                stability = "Stable"
        else:
            stability = "Unknown"

        result = {
            "brightness": round(brightness, 2),
            "light_level": light_level,
            "stability": stability
        }

        if overlay:
            self._draw_overlay(frame, result)

        return result, frame

    def _draw_overlay(self, frame, result):
        h, w, _ = frame.shape

        text1 = f"Brightness: {result['brightness']}"
        text2 = f"Light: {result['light_level']}"
        text3 = f"Stability: {result['stability']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        color = (0, 255, 0)
        color_warn = (0, 0, 255)

        c2 = color_warn if result["light_level"] != "Normal" else color
        c3 = color_warn if result["stability"] != "Stable" else color

        def get_text_pos(text, y_offset):
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            x = w - text_width - 10   
            y = y_offset
            return x, y

        cv2.putText(frame, text1, get_text_pos(text1, 25), font, scale, color, thickness)
        cv2.putText(frame, text2, get_text_pos(text2, 55), font, scale, c2, thickness)
        cv2.putText(frame, text3, get_text_pos(text3, 85), font, scale, c3, thickness)