import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
from PIL import Image
import threading
import time
from typing import List, Tuple, Optional


class ScreenCapture:
    """
    Klasse für die Bildschirmaufnahme von Windows-Anwendungen
    """
    
    def __init__(self):
        self.window = None
        self.is_capturing = False
        self.capture_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def get_available_windows(self) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren Fenster zurück
        """
        windows = gw.getAllWindows()
        window_titles = []
        for window in windows:
            if window.title and window.title.strip() and window.visible:
                # Filtere System-Fenster aus
                if not any(sys_window in window.title.lower() for sys_window in 
                          ['desktop', 'taskbar', 'start menu', 'cortana', 'windows input']):
                    window_titles.append(window.title)
        
        return sorted(list(set(window_titles)))
    
    def select_window(self, window_title: str) -> bool:
        """
        Wählt ein Fenster für die Aufnahme aus
        """
        try:
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                self.window = windows[0]
                # Fenster in den Vordergrund bringen
                self.window.activate()
                time.sleep(0.5)  # Kurz warten bis Fenster aktiv ist
                print(f"Window selected: {window_title} (Handle: {self.window._hWnd})")
                return True
        except Exception as e:
            print(f"Fehler beim Auswählen des Fensters: {e}")
        
        return False
    
    def get_window_handle(self):
        """
        Gibt den Windows-Handle des ausgewählten Fensters zurück
        """
        if self.window and hasattr(self.window, '_hWnd'):
            return self.window._hWnd
        return None
    
    def start_capture(self) -> bool:
        """
        Startet die kontinuierliche Bildschirmaufnahme
        """
        if not self.window:
            print("Kein Fenster ausgewählt!")
            return False
            
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        return True
    
    def stop_capture(self):
        """
        Stoppt die Bildschirmaufnahme
        """
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
    
    def _capture_loop(self):
        """
        Interne Methode für die kontinuierliche Aufnahme
        """
        while self.is_capturing:
            try:
                if self.window and self.window.visible:
                    # Screenshot des Fensters machen
                    screenshot = pyautogui.screenshot(region=(
                        self.window.left,
                        self.window.top,
                        self.window.width,
                        self.window.height
                    ))
                    
                    # In OpenCV Format konvertieren
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    
                    with self.frame_lock:
                        self.current_frame = frame
                
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"Fehler bei der Aufnahme: {e}")
                time.sleep(0.1)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Gibt den aktuellen Frame zurück
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_processed_frame(self, target_size: Tuple[int, int] = (84, 84)) -> Optional[np.ndarray]:
        """
        Gibt einen für die AI vorverarbeiteten Frame zurück
        """
        frame = self.get_current_frame()
        if frame is None:
            return None
            
        # In Graustufen konvertieren
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Größe ändern für AI-Input
        resized_frame = cv2.resize(gray_frame, target_size)
        
        # Normalisieren (0-1)
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        return normalized_frame
    
    def get_window_info(self) -> dict:
        """
        Gibt Informationen über das ausgewählte Fenster zurück
        """
        if not self.window:
            return {}
            
        return {
            'title': self.window.title,
            'left': self.window.left,
            'top': self.window.top,
            'width': self.window.width,
            'height': self.window.height,
            'visible': self.window.visible
        }
