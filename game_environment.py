import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional
from screen_capture import ScreenCapture
from game_controller import GameController, Action
from ai_agent import DQNAgent
import time


class GameEnvironment:
    """
    Spielumgebung für das Reinforcement Learning
    """
    
    def __init__(self, screen_capture: ScreenCapture, game_controller: GameController,
                 frame_stack_size: int = 4, action_repeat: int = 4):
        self.screen_capture = screen_capture
        self.game_controller = game_controller
        self.frame_stack_size = frame_stack_size
        self.action_repeat = action_repeat
        
        # Frame Stack für zeitliche Informationen
        self.frame_stack = deque(maxlen=frame_stack_size)
        
        # Reward System
        self.last_x_position = 0
        self.max_x_position = 0
        self.frames_since_progress = 0
        self.max_frames_without_progress = 300  # ~10 Sekunden bei 30 FPS
        
        # Episode Tracking
        self.episode_length = 0
        self.episode_reward = 0
        
        # Game State Detection
        self.last_frame = None
        self.death_detection_threshold = 0.95  # Ähnlichkeit für Toddetektion
        
    def reset(self) -> np.ndarray:
        """
        Setzt die Umgebung zurück für eine neue Episode
        """
        # Frame Stack leeren und mit ersten Frames füllen
        self.frame_stack.clear()
        
        # Warten bis gültiger Frame verfügbar ist
        current_frame = None
        for _ in range(30):  # Max 1 Sekunde warten
            current_frame = self.screen_capture.get_processed_frame()
            if current_frame is not None:
                break
            time.sleep(1/30)
        
        if current_frame is None:
            # Fallback: Schwarzer Frame
            current_frame = np.zeros((84, 84), dtype=np.float32)
        
        # Frame Stack mit identischen Frames initialisieren
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(current_frame)
          # Reset game state
        self.last_x_position = 0
        self.max_x_position = 0
        self.frames_since_progress = 0
        self.episode_length = 0
        self.episode_reward = 0
        self.last_frame = current_frame.copy()
        
        # Episode-Counter erhöhen (für Debug-Ausgaben)
        if not hasattr(self, 'current_episode'):
            self.current_episode = 1
        else:
            self.current_episode += 1
        
        print(f"\n🆕 [Episode {self.current_episode}] Starting new episode...")
        
        # Alle Tasten loslassen
        self.game_controller.perform_action(Action.NOTHING)
        
        return self._get_state()
    
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Führt eine Aktion aus und gibt den neuen Zustand zurück
        """
        action = Action(action_index)
        
        # Aktion mehrfach wiederholen für bessere Kontrolle
        total_reward = 0
        
        for _ in range(self.action_repeat):
            self.game_controller.perform_action(action)
            time.sleep(1/60)  # Kurz warten zwischen Wiederholungen
            
            # Neuen Frame holen
            new_frame = self.screen_capture.get_processed_frame()
            if new_frame is not None:
                self.frame_stack.append(new_frame)
                self.last_frame = new_frame.copy()
            
            # Reward berechnen
            reward = self._calculate_reward(action)
            total_reward += reward
            
            self.episode_length += 1
            self.frames_since_progress += 1
        
        # Episode beenden wenn nötig
        done = self._check_episode_end()
        
        self.episode_reward += total_reward
        
        info = {
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'max_x_position': self.max_x_position,
            'frames_since_progress': self.frames_since_progress
        }
        
        return self._get_state(), total_reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Gibt den aktuellen Zustand als gestapelte Frames zurück
        """
        if len(self.frame_stack) < self.frame_stack_size:
            # Fallback: Fülle mit Nullen auf
            frames = list(self.frame_stack)
            while len(frames) < self.frame_stack_size:
                frames.insert(0, np.zeros((84, 84), dtype=np.float32))
            return np.stack(frames, axis=0)
        
        return np.stack(self.frame_stack, axis=0)
    def _calculate_reward(self, action: Action) -> float:
        """
        Berechnet den Reward für die aktuelle Aktion
        """
        reward = 0.0
        debug_messages = []
        
        # Fortschritts-Reward (Bewegung nach rechts)
        current_x = self._estimate_x_position()
        
        if current_x > self.max_x_position:
            # Neue maximale Position erreicht
            progress_reward = (current_x - self.max_x_position) * 10.0  # Belohnung für Fortschritt
            reward += progress_reward
            self.max_x_position = current_x
            self.frames_since_progress = 0
            debug_messages.append(f"🏃 Reward Gained (Right Movement Detected): +{progress_reward:.2f}")
        
        # Aktions-Rewards
        if action == Action.RIGHT:
            reward += 0.1  # Kleine Belohnung für Rechtsbewegung
            debug_messages.append("➡️ Reward Gained (Right Action): +0.10")
        elif action == Action.LEFT:
            reward -= 0.1  # Kleine Strafe für Linksbewegung
            debug_messages.append("⬅️ Punishment (Left Movement Detected): -0.10")
        elif action == Action.JUMP:
            reward += 0.05  # Kleine Belohnung für Springen (kann nützlich sein)
            debug_messages.append("🦘 Reward Gained (Jump): +0.05")
        
        # Strafe für Stillstand
        if self.frames_since_progress > 60:  # 2 Sekunden ohne Fortschritt
            reward -= 0.5
            debug_messages.append("⏰ Punishment for no movement: -0.50")
          # Lebens-Reward (kleine kontinuierliche Belohnung für Überleben)
        reward += 0.01
        debug_messages.append("💖 Life Reward: +0.01")
        
        # Death Screen Strafe
        if len(self.frame_stack) > 0:
            current_frame = self.frame_stack[-1]
            if self._is_death_screen(current_frame):
                reward -= 10.0  # Große Strafe für Death Screen
                debug_messages.append("💀 Death Screen Punishment: -10.00")
        
        # Debug-Ausgaben anzeigen (nur wenn signifikante Rewards/Punishments)
        significant_messages = [msg for msg in debug_messages if not msg.startswith("💖")]
        if significant_messages:
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] Step {self.episode_length}: " + 
                  " | ".join(significant_messages) + f" | Total: {reward:.3f}")
        
        return reward
    
    def _estimate_x_position(self) -> float:
        """
        Schätzt die X-Position des Spielers basierend auf dem aktuellen Frame
        Einfache Heuristik: Mittlere Helligkeit des rechten Bildschirmbereichs
        """
        if len(self.frame_stack) == 0:
            return 0.0
        
        current_frame = self.frame_stack[-1]
        
        # Verwende verschiedene Bereiche des Bildes als Proxy für Position
        # Dies ist eine vereinfachte Heuristik - in einem echten Spiel könnte man
        # Template Matching oder andere Computer Vision Techniken verwenden
        
        # Analysiere rechte Hälfte des Bildes
        right_half = current_frame[:, 42:]
        
        # Verwende Gradientenanalyse als Proxy für Bewegung/Position
        gradient_x = np.gradient(right_half, axis=1)
        position_estimate = np.sum(np.abs(gradient_x)) / 1000.0  # Normalisierung
        
        return position_estimate
    def _check_episode_end(self) -> bool:
        """
        Überprüft ob die Episode beendet werden soll
        """
        # Episode zu lang
        if self.episode_length > 5000:  # Max ~3 Minuten bei 30 FPS
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] 🕐 Episode ended: Maximum length reached (5000 steps)")
            return True
        
        # Zu lange ohne Fortschritt
        if self.frames_since_progress > self.max_frames_without_progress:
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] ⏰ Episode ended: No progress for {self.max_frames_without_progress} frames")
            return True
        
        # Tod-Detection (vereinfacht durch Frame-Vergleich)
        if self._detect_death():
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] 💀 Punishment for dead: Episode ended due to death detection")
            return True
        
        return False
    def _detect_death(self) -> bool:
        """
        Erweiterte Toddetektion durch Erkennung von schwarzen Bildschirmen und Bildvergleich
        """
        if len(self.frame_stack) < 2:
            return False
        
        current_frame = self.frame_stack[-1]
        
        # 1. Death Screen Erkennung: Überprüfe ob Bildschirm überwiegend schwarz ist
        if self._is_death_screen(current_frame):
            return True
        
        # 2. Plötzliche Bildveränderung (ursprüngliche Methode)
        previous_frame = self.frame_stack[-2]
        diff = np.abs(current_frame - previous_frame)
        
        # Wenn sich das Bild plötzlich stark ändert, könnte es ein Tod sein
        if np.mean(diff) > 0.3:  # Threshold für starke Veränderung
            return True
        
        return False
    
    def _is_death_screen(self, frame: np.ndarray, black_threshold: float = 0.15, 
                        black_percentage_threshold: float = 0.75) -> bool:
        """
        Überprüft ob der aktuelle Frame ein Death Screen ist (überwiegend schwarz)
        
        Args:
            frame: Der zu überprüfende Frame (normalisiert 0-1)
            black_threshold: Schwellenwert für "schwarze" Pixel (0-1)
            black_percentage_threshold: Mindestprozentsatz schwarzer Pixel für Death Screen
        
        Returns:
            True wenn Death Screen erkannt wurde
        """
        # Zähle Pixel die unter dem Schwarz-Threshold liegen
        black_pixels = np.sum(frame < black_threshold)
        total_pixels = frame.size
        black_percentage = black_pixels / total_pixels
        
        # Debug-Ausgabe für Entwicklung (kann später entfernt werden)
        if black_percentage > 0.5:  # Nur bei hohem Schwarzanteil ausgeben
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] 🖥️ Black screen detection: {black_percentage:.2%} black pixels")
        
        # Death Screen wenn überwiegend schwarz
        if black_percentage >= black_percentage_threshold:
            episode_num = getattr(self, 'current_episode', 'N/A')
            print(f"[Episode {episode_num}] 💀 Death Screen detected: {black_percentage:.2%} black pixels")
            return True
        
        return False
    
    def get_frame_for_display(self) -> Optional[np.ndarray]:
        """
        Gibt den aktuellen Frame für die Anzeige zurück
        """
        if len(self.frame_stack) > 0:
            # Konvertiere zurück zu 0-255 für Anzeige
            frame = (self.frame_stack[-1] * 255).astype(np.uint8)
            # Zu RGB für bessere Anzeige
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return None
    
    def test_death_screen_detection(self, test_frames: int = 100):
        """
        Testet die Death Screen-Erkennung mit den aktuellen Frames
        """
        print("=== DEATH SCREEN DETECTION TEST ===")
        print(f"Teste Death Screen-Erkennung für {test_frames} Frames...")
        
        death_screens_detected = 0
        
        for i in range(test_frames):
            current_frame = self.screen_capture.get_processed_frame()
            if current_frame is not None:
                is_death = self._is_death_screen(current_frame)
                if is_death:
                    death_screens_detected += 1
                    print(f"Frame {i+1}: 💀 Death Screen detected!")
                elif i % 20 == 0:  # Alle 20 Frames Status ausgeben
                    black_pixels = np.sum(current_frame < 0.15)
                    total_pixels = current_frame.size
                    black_percentage = black_pixels / total_pixels
                    print(f"Frame {i+1}: ✅ Normal ({black_percentage:.1%} black)")
            
            time.sleep(0.1)  # Kurz warten zwischen Tests
        
        print(f"\n✅ Test beendet: {death_screens_detected}/{test_frames} Death Screens erkannt")
        return death_screens_detected

    def cleanup(self):
        """
        Aufräumen am Ende
        """
        self.game_controller.cleanup()
