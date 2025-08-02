import threading
import time
import os
from datetime import datetime
from typing import Optional
from screen_capture import ScreenCapture
from game_controller import GameController
from ai_agent import DQNAgent
from game_environment import GameEnvironment


class TrainingManager:
    """
    Manager für das Training der AI
    """
    
    def __init__(self, screen_capture: ScreenCapture, save_dir: str = "models"):
        self.screen_capture = screen_capture
        self.save_dir = save_dir
        
        # Erstelle Modell-Verzeichnis falls nicht vorhanden
        os.makedirs(save_dir, exist_ok=True)
          # Komponenten
        self.game_controller = GameController()
        self.environment = GameEnvironment(screen_capture, self.game_controller)
        self.agent = DQNAgent()
        
        # Fenster-Handle für bessere Eingabeübertragung setzen
        window_handle = screen_capture.get_window_handle()
        if window_handle:
            self.game_controller.set_game_window(window_handle)
            print(f"Game controller configured with window handle: {window_handle}")
        else:
            print("Warning: No window handle available for direct input")
        
        # Training State
        self.is_training = False
        self.training_thread = None
        self.current_episode = 0
        self.max_episodes = 1000
        
        # Test State
        self.is_testing = False
        self.test_thread = None
        
        # Statistics
        self.training_start_time = None
        self.last_save_time = 0
        self.save_interval = 300  # Speichere alle 5 Minuten
        
    def start_training(self, max_episodes: int = 1000, load_existing: bool = True):
        """
        Startet das Training
        """
        if self.is_training:
            print("Training läuft bereits!")
            return False
        
        self.max_episodes = max_episodes
        
        # Versuche existierendes Modell zu laden
        if load_existing:
            model_path = self._get_latest_model_path()
            if model_path and os.path.exists(model_path):
                self.agent.load_model(model_path)
                print(f"Bestehendes Modell geladen: {model_path}")
        
        self.is_training = True
        self.training_start_time = time.time()
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        print(f"Training gestartet für {max_episodes} Episoden...")
        return True
    
    def stop_training(self):
        """
        Stoppt das Training
        """
        if not self.is_training:
            print("Training läuft nicht!")
            return
        
        self.is_training = False
        if self.training_thread:
            self.training_thread.join()
        
        # Modell speichern
        self._save_model()
        print("Training gestoppt und Modell gespeichert.")
    
    def _training_loop(self):
        """
        Haupt-Trainingsschleife
        """
        try:
            episode = len(self.agent.episode_rewards)
            
            while self.is_training and episode < self.max_episodes:
                episode_start_time = time.time()
                
                # Episode zurücksetzen
                state = self.environment.reset()
                episode_reward = 0
                episode_length = 0
                
                print(f"\nEpisode {episode + 1}/{self.max_episodes} gestartet...")
                
                while self.is_training:
                    # Aktion auswählen
                    action = self.agent.select_action(state, training=True)
                    
                    # Schritt in der Umgebung
                    next_state, reward, done, info = self.environment.step(action)
                    
                    # Erfahrung speichern
                    self.agent.store_experience(state, action, reward, next_state, done)
                    
                    # Training step
                    loss = self.agent.train_step()
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                # Episode Statistiken
                self.agent.episode_rewards.append(episode_reward)
                self.agent.episode_lengths.append(episode_length)
                
                episode_duration = time.time() - episode_start_time
                
                print(f"Episode {episode + 1} beendet:")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Länge: {episode_length} Steps")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Dauer: {episode_duration:.1f}s")
                print(f"  Max X Position: {info.get('max_x_position', 0):.2f}")
                
                # Periodisches Speichern
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_model()
                    self.last_save_time = current_time
                
                episode += 1
            
            # Training beendet
            self._save_model()
            print(f"\nTraining nach {episode} Episoden beendet!")
            
        except Exception as e:
            print(f"Fehler beim Training: {e}")
        finally:
            self.is_training = False
    
    def _save_model(self):
        """
        Speichert das aktuelle Modell
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mario_ai_model_{timestamp}.pth"
        filepath = os.path.join(self.save_dir, filename)
        self.agent.save_model(filepath)
        
        # Erstelle auch eine "latest" Kopie
        latest_path = os.path.join(self.save_dir, "mario_ai_model_latest.pth")
        self.agent.save_model(latest_path)
    
    def _get_latest_model_path(self) -> Optional[str]:
        """
        Gibt den Pfad zum neuesten Modell zurück
        """
        latest_path = os.path.join(self.save_dir, "mario_ai_model_latest.pth")
        if os.path.exists(latest_path):
            return latest_path
        
        # Suche nach anderen Modellen
        if os.path.exists(self.save_dir):
            model_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
            if model_files:
                # Sortiere nach Änderungszeit
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)), reverse=True)
                return os.path.join(self.save_dir, model_files[0])
        
        return None
    
    def test_agent(self, episodes: int = 5, model_path: Optional[str] = None):
        """
        Testet den trainierten Agent
        """
        if self.is_testing:
            print("Agent-Test läuft bereits!")
            return
            
        self.is_testing = True
        self.test_thread = threading.Thread(target=self._test_agent_loop, args=(episodes, model_path))
        self.test_thread.daemon = True
        self.test_thread.start()
    
    def _test_agent_loop(self, episodes: int, model_path: Optional[str] = None):
        """
        Führt den Agent-Test in einem separaten Thread aus
        """
        try:
            if model_path is None:
                model_path = self._get_latest_model_path()
            
            if model_path and os.path.exists(model_path):
                self.agent.load_model(model_path)
                print(f"Modell für Test geladen: {model_path}")
            else:
                print("Kein Modell zum Testen gefunden!")
                return
            
            print(f"\nTeste Agent für {episodes} Episoden...")
            
            for episode in range(episodes):
                if not self.is_testing:  # Check if test was stopped
                    break
                    
                state = self.environment.reset()
                episode_reward = 0
                episode_length = 0
                
                print(f"Test Episode {episode + 1}...")
                
                while self.is_testing:
                    # Aktion ohne Exploration auswählen
                    action = self.agent.select_action(state, training=False)
                    
                    # Schritt in der Umgebung
                    state, reward, done, info = self.environment.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                print(f"  Reward: {episode_reward:.2f}, Länge: {episode_length}, Max X: {info.get('max_x_position', 0):.2f}")
                
        except Exception as e:
            print(f"Fehler beim Agent-Test: {e}")
        finally:
            self.is_testing = False
    
    def stop_agent_test(self):
        """
        Stoppt den Agent-Test
        """
        if not self.is_testing:
            print("Agent-Test läuft nicht!")
            return
            
        self.is_testing = False
        if self.test_thread:
            self.test_thread.join()
        print("Agent-Test gestoppt.")
    
    def load_latest_model(self) -> bool:
        """
        Lädt das neueste verfügbare Modell
        """
        model_path = self._get_latest_model_path()
        
        if model_path and os.path.exists(model_path):
            success = self.agent.load_model(model_path)
            if success:
                print(f"Neuestes Modell geladen: {model_path}")
                return True
            else:
                print(f"Fehler beim Laden des Modells: {model_path}")
                return False
        else:
            print("Kein gespeichertes Modell gefunden!")
            return False
    
    def get_training_stats(self) -> dict:
        """
        Gibt aktuelle Trainingsstatistiken zurück
        """
        stats = self.agent.get_stats()
        
        if self.training_start_time:
            training_duration = time.time() - self.training_start_time
            stats['training_duration'] = training_duration
            stats['episodes_per_hour'] = len(self.agent.episode_rewards) / (training_duration / 3600) if training_duration > 0 else 0
        
        stats['is_training'] = self.is_training
        stats['current_episode'] = len(self.agent.episode_rewards)
        stats['max_episodes'] = self.max_episodes
        
        return stats
    
    def cleanup(self):
        """
        Aufräumen
        """
        if self.is_training:
            self.stop_training()
        self.environment.cleanup()
