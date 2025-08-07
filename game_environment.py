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
          # Leben-System (3 Leben pro Game)
        self.lives_remaining = 3
        self.total_deaths = 0  # Gesamt-Tode für Statistiken
        self.death_screen_start_time = None
        self.waiting_after_death = False
        self.game_over_state = False
        self.waiting_for_game_restart = False
        
        # Start Screen Protection (nach Game Over Restart)
        self.game_just_restarted = False
        self.restart_protection_start_time = None
        self.restart_protection_duration = 10.0  # 10 Sekunden Death Detection deaktiviert
    def reset(self) -> np.ndarray:
        """
        Setzt die Umgebung zurück für eine neue Episode
        """
        # Nur bei echtem Reset (nicht nach Tod) Leben zurücksetzen
        if not self.waiting_after_death and not self.waiting_for_game_restart:
            self.lives_remaining = 3
            self.total_deaths = 0
            self.game_over_state = False
            print(f"\n🆕 [Episode {getattr(self, 'current_episode', 1)}] Starting fresh game with 3 lives...")
        
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
          # Death Screen States zurücksetzen
        self.death_screen_start_time = None
        self.waiting_after_death = False
        self.waiting_for_game_restart = False
        
        # Start Screen Protection NICHT zurücksetzen - soll aktiv bleiben nach Game Over Recovery!
        
        # Episode-Counter erhöhen (für Debug-Ausgaben)
        if not hasattr(self, 'current_episode'):
            self.current_episode = 1
        else:
            self.current_episode += 1
        
        print(f"🆕 [Episode {self.current_episode}] Starting - Lives: {self.lives_remaining}/3")
        
        # Alle Tasten loslassen
        self.game_controller.perform_action(Action.NOTHING)
        
        return self._get_state()
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Führt eine Aktion aus und gibt den neuen Zustand zurück
        """
        # Prüfe ob wir in einem Death Screen warten müssen
        if self.waiting_after_death or self.waiting_for_game_restart:
            return self._handle_death_recovery()
        
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
            'frames_since_progress': self.frames_since_progress,
            'lives_remaining': self.lives_remaining,
            'total_deaths': self.total_deaths
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

        # Tod-Detection: Nur wenn keine Recovery läuft
        if not self.waiting_after_death and not self.waiting_for_game_restart:
            if self._detect_death():
                return self._handle_death_detected()
        
        return False
    
    def _handle_death_detected(self) -> bool:
        """
        Behandelt den Tod: Setzt Recovery-Flags und startet Recovery, beendet aber NICHT sofort die Episode!
        """
        episode_num = getattr(self, 'current_episode', 'N/A')
        self.total_deaths += 1
        print(f"💀 [Episode {episode_num}] Death detected!")
        if self.lives_remaining > 1:
            self.lives_remaining -= 1
            self.waiting_after_death = True
            self.death_screen_start_time = time.time()
        else:
            self.lives_remaining = 0
            self.waiting_for_game_restart = True
            self.death_screen_start_time = time.time()        # NICHT sofort die Episode beenden!
        return False
    
    def _detect_death(self) -> bool:
        """
        Erweiterte Toddetektion durch Erkennung von schwarzen Bildschirmen und Bildvergleich
        """
        # Während Recovery keine Death Detection!
        if self.waiting_after_death or self.waiting_for_game_restart:
            return False
            
        # Nach Game Over Restart Schutz - keine Death Detection für eine Weile
        if self.game_just_restarted:
            current_time = time.time()
            if self.restart_protection_start_time is None:
                self.restart_protection_start_time = current_time
                elapsed_protection_time = current_time - self.restart_protection_start_time
            if elapsed_protection_time < self.restart_protection_duration:
                # Noch im Schutz-Modus - Debug-Ausgabe für bessere Nachverfolgung
                episode_num = getattr(self, 'current_episode', 'N/A')
                print(f"🛡️ [Episode {episode_num}] Start screen protection active - ignoring black screen ({elapsed_protection_time:.1f}s/{self.restart_protection_duration}s)")
                return False
            else:
                # Schutz-Modus beenden
                print(f"🛡️ [Episode {getattr(self, 'current_episode', 'N/A')}] Start screen protection ended after {elapsed_protection_time:.1f}s")
                self.game_just_restarted = False
                self.restart_protection_start_time = None
        
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
    
    def _handle_death_recovery(self) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Behandelt die Recovery nach einem Death Screen
        """
        # Aktuellen Frame holen
        current_frame = self.screen_capture.get_processed_frame()
        if current_frame is not None:
            self.frame_stack.append(current_frame)
        
        # Prüfe ob wir nach Game Over warten (3 Tode erreicht)
        if self.waiting_for_game_restart:
            return self._handle_game_over_recovery()
        
        # Prüfe ob wir nach einem einzelnen Tod warten
        if self.waiting_after_death:
            return self._handle_single_death_recovery()
        
        # Fallback
        return self._get_state(), 0.0, False, {
            'lives_remaining': self.lives_remaining,
            'total_deaths': self.total_deaths,
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward
        }
    
    def _handle_single_death_recovery(self) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Behandelt Recovery nach einem einzelnen Tod (noch Leben übrig)
        """
        current_time = time.time()
        current_frame = self.frame_stack[-1] if len(self.frame_stack) > 0 else None

        # Prüfe ob der Death Screen vorbei ist (nicht mehr überwiegend schwarz)
        if current_frame is not None and not self._is_death_screen(current_frame):
            # Death Screen ist vorbei, jetzt explizit 3 Sekunden warten
            if not hasattr(self, '_single_death_wait_start') or self._single_death_wait_start is None:
                self._single_death_wait_start = current_time
                print(f"🕒 [Episode {self.current_episode}] Death screen cleared - waiting 3s before continue")
                return self._get_state(), 0.0, False, {
                    'lives_remaining': self.lives_remaining,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
            elif current_time - self._single_death_wait_start < 3.0:
                # Noch nicht genug gewartet
                return self._get_state(), 0.0, False, {
                    'lives_remaining': self.lives_remaining,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
            else:
                # 3 Sekunden sind vorbei, weiterspielen
                print(f"💚 [Episode {self.current_episode}] Waited 3s after death screen - Continuing with {self.lives_remaining} lives")
                self.waiting_after_death = False
                self.death_screen_start_time = None
                self._single_death_wait_start = None
                self.last_x_position = 0
                self.max_x_position = 0
                self.frames_since_progress = 0
                return self._get_state(), 0.1, False, {
                    'lives_remaining': self.lives_remaining,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }

        # Immer noch im Death Screen - weiter warten
        elapsed_time = current_time - self.death_screen_start_time
        if elapsed_time > 10.0:  # Max 10 Sekunden warten
            print(f"⚠️ [Episode {self.current_episode}] Death screen timeout after 10s - forcing continue")
            self.waiting_after_death = False
            self.death_screen_start_time = None
            self._single_death_wait_start = None
        return self._get_state(), 0.0, False, {
            'lives_remaining': self.lives_remaining,
            'total_deaths': self.total_deaths,
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward
        }
    
    def _handle_game_over_recovery(self) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Behandelt Recovery nach Game Over (alle 3 Leben verloren)
        """
        current_time = time.time()
        current_frame = self.frame_stack[-1] if len(self.frame_stack) > 0 else None
        elapsed_time = current_time - self.death_screen_start_time if self.death_screen_start_time else 0

        # Phase 1: Warten bis Screen nicht mehr schwarz ist
        if current_frame is not None and self._is_death_screen(current_frame):
            if not hasattr(self, '_game_over_wait_phase') or self._game_over_wait_phase != 1:
                self._game_over_wait_phase = 1
                self._game_over_wait_start = None
            return self._get_state(), 0.0, False, {
                'lives_remaining': 0,
                'total_deaths': self.total_deaths,
                'episode_length': self.episode_length,
                'episode_reward': self.episode_reward
            }        # Phase 2: Screen ist nicht mehr schwarz - warte 5 Sekunden, dann ENTER
        if (not hasattr(self, '_game_over_wait_phase') or self._game_over_wait_phase == 1) and current_frame is not None and not self._is_death_screen(current_frame):
            if not hasattr(self, '_game_over_wait_start') or self._game_over_wait_start is None:
                self._game_over_wait_start = current_time
                self._game_over_wait_phase = 2  # <--- Fix: set phase to 2 here
                print(f"🕒 [Episode {self.current_episode}] Game over screen cleared - waiting 5s before ENTER")
                return self._get_state(), 0.0, False, {
                    'lives_remaining': 0,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
        # Phase 2: Warten bis 5 Sekunden vorbei, dann ENTER
        if hasattr(self, '_game_over_wait_phase') and self._game_over_wait_phase == 2:
            if current_time - self._game_over_wait_start < 5.0:
                return self._get_state(), 0.0, False, {
                    'lives_remaining': 0,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
            else:
                print(f"🔄 [Episode {self.current_episode}] Waited 5s - pressing 'i' to restart")
                self.game_controller.press_key('i')
                self._game_over_wait_phase = 3
                self._game_over_wait_start = current_time
                return self._get_state(), 0.0, False, {
                    'lives_remaining': 0,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
        # Phase 3: Nach ENTER 3-4 Sekunden warten, dann neues Spiel
        if hasattr(self, '_game_over_wait_phase') and self._game_over_wait_phase == 3:
            if current_time - self._game_over_wait_start < 3.5:                return self._get_state(), 0.0, False, {
                    'lives_remaining': 0,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward
                }
            else:
                print(f"🎮 [Episode {self.current_episode}] Waited 3.5s after ENTER - New game started - Episode completed")
                
                # START SCREEN SCHUTZ AKTIVIEREN
                self.game_just_restarted = True
                self.restart_protection_start_time = time.time()
                print(f"🛡️ [Episode {self.current_episode}] Start screen protection activated for {self.restart_protection_duration}s")
                
                self.waiting_for_game_restart = False
                self.death_screen_start_time = None
                self._game_over_wait_phase = None
                self._game_over_wait_start = None
                return self._get_state(), 0.0, True, {
                    'lives_remaining': 3,
                    'total_deaths': self.total_deaths,
                    'episode_length': self.episode_length,
                    'episode_reward': self.episode_reward,
                    'game_over_recovery': True
                }
        # Fallback
        return self._get_state(), 0.0, False, {
            'lives_remaining': 0,
            'total_deaths': self.total_deaths,
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward
        }
