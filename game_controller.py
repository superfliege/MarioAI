import pyautogui
import keyboard
import time
from typing import List
from enum import Enum


class Action(Enum):
    """
    Enum für mögliche Aktionen im Spiel
    """
    NOTHING = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    DOWN = 4


class GameController:
    """
    Klasse für die Spielsteuerung mit simulierten Keyboard-Aktionen
    """
    
    def __init__(self):
        # Automatische Failsafe deaktivieren für bessere Performance
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01  # Minimale Pause zwischen Aktionen
          # Mapping von Aktionen auf Tasten
        self.action_keys = {
            Action.LEFT: 'a',     # Links mit Taste A
            Action.RIGHT: 'd',    # Rechts mit Taste D
            Action.JUMP: 'v',     # Springen mit Taste V
            Action.DOWN: 'down'   # Down bleibt Pfeiltaste (falls benötigt)
        }
        
        self.last_action = Action.NOTHING
        self.key_press_duration = 0.1
        
    def set_game_window(self, window_handle):
        """
        Setzt das Spielfenster (nicht mehr benötigt für simulierte Eingaben)
        """
        # Diese Methode wird für Kompatibilität beibehalten, macht aber nichts
        # da wir nur simulierte Eingaben verwenden
        pass
    
    def perform_action(self, action: Action):
        """
        Führt eine Aktion im Spiel aus mit simulierten Eingaben
        """
        # Vorherige Taste loslassen wenn nötig
        if self.last_action != Action.NOTHING and self.last_action != action:
            self._release_key(self.last_action)
        
        if action == Action.NOTHING:
            # Alle Tasten loslassen
            self._release_all_keys()
        else:
            # Neue Taste drücken
            self._press_key(action)
        
        self.last_action = action
    
    def _press_key(self, action: Action):
        """
        Drückt eine Taste mit simulierten Keyboard-Aktionen
        """
        if action in self.action_keys:
            key = self.action_keys[action]
            
            try:
                # Simulierte Tasteneingabe mit PyAutoGUI
                pyautogui.keyDown(key)
                
                # Zusätzlich Keyboard Library für bessere Kompatibilität
                try:
                    keyboard.press(key)
                except:
                    pass  # Ignoriere Fehler
                        
            except Exception as e:
                print(f"❌ Error pressing key {action.name}: {e}")
                # Fallback nur mit PyAutoGUI
                try:
                    pyautogui.keyDown(key)
                except Exception as final_e:
                    print(f"❌ All key press methods failed: {final_e}")
    
    def _release_key(self, action: Action):
        """
        Lässt eine Taste los mit simulierten Keyboard-Aktionen
        """
        if action in self.action_keys:
            key = self.action_keys[action]
            
            try:
                # Simulierte Tasteneingabe mit PyAutoGUI
                pyautogui.keyUp(key)
                
                # Zusätzlich Keyboard Library für bessere Kompatibilität
                try:
                    keyboard.release(key)
                except:
                    pass  # Ignoriere Fehler
                        
            except Exception as e:
                print(f"❌ Error releasing key {action.name}: {e}")
                # Fallback nur mit PyAutoGUI
                try:
                    pyautogui.keyUp(key)
                except Exception as final_e:
                    print(f"❌ All key release methods failed: {final_e}")
    
    def _release_all_keys(self):
        """
        Lässt alle Spieltasten los
        """
        for action in self.action_keys:
            self._release_key(action)
    
    def get_available_actions(self) -> List[Action]:
        """
        Gibt alle verfügbaren Aktionen zurück
        """
        return list(Action)
    
    def action_to_index(self, action: Action) -> int:
        """
        Konvertiert eine Aktion in einen Index
        """
        return action.value
    
    def index_to_action(self, index: int) -> Action:
        """
        Konvertiert einen Index in eine Aktion
        """
        return Action(index)
    
    def test_keys(self):
        """
        Testet alle Tasten für 2 Sekunden jede
        """        
        print("=== TASTENEINGABE TEST ===")
        print("Teste alle Aktionen für je 2 Sekunden...")
        print("A=Links, D=Rechts, V=Springen")
        
        actions_to_test = [Action.RIGHT, Action.LEFT, Action.JUMP, Action.DOWN]
        
        for action in actions_to_test:
            key = self.action_keys[action]
            print(f"\nTeste {action.name} ({key}) für 2 Sekunden...")
            
            # Taste drücken
            self.perform_action(action)
            time.sleep(2)
            
            # Taste loslassen
            self.perform_action(Action.NOTHING)
            time.sleep(0.5)
        
        print("\nTastentest beendet!")
    
    def test_input_methods(self):
        """
        Testet simulierte Keyboard-Eingaben
        """
        print("=== SIMULIERTE KEYBOARD-EINGABEN TEST ===")
        print("Teste verschiedene Tasten mit simulierten Eingaben...")
        
        # Test PyAutoGUI Simulationen
        print("\n🔸 Test 1: PyAutoGUI Simulation")
        print("Drücke D (RECHTS) für 1 Sekunde...")
        pyautogui.keyDown('d')
        time.sleep(1)
        pyautogui.keyUp('d')
        time.sleep(0.5)
        print("\n🔸 Test 2: Keyboard Library Simulation")
        print("Drücke A (LINKS) für 1 Sekunde...")
        try:
            keyboard.press('a')
            time.sleep(1)
            keyboard.release('a')
        except Exception as e:
            print(f"Keyboard Library Fehler: {e}")
        time.sleep(0.5)
        
        # Kombinierter Test
        print("\n🔸 Test 3: Kombinierte Simulation")
        print("Drücke V (SPRUNG) für 1 Sekunde...")
        self.perform_action(Action.JUMP)
        time.sleep(1)
        self.perform_action(Action.NOTHING)
        
        print("\n✅ Simulierte Eingaben Test beendet!")
    
    def press_enter(self):
        """
        Drückt die ENTER-Taste (für Game Over Recovery)
        """
        try:
            print("🔄 Pressing ENTER key...")
            pyautogui.press('enter')
            time.sleep(0.1)
            
            # Zusätzlich mit keyboard library versuchen
            try:
                keyboard.press_and_release('enter')
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error pressing ENTER: {e}")
    def cleanup(self):
        """
        Räumt auf und lässt alle Tasten los
        """
        self._release_all_keys()
        pyautogui.FAILSAFE = True  # Failsafe wieder aktivieren
    def press_key(self, key: str):
        """
        Drückt eine beliebige Taste (für Game Over Recovery etc.)
        Optimiert für Mario-Spiel mit mehreren Fallback-Mechanismen
        """
        try:
            print(f"🔄 Pressing key '{key}'...")
            
            # Spezielle Behandlung für 'i' Taste (Mario restart)
            if key.lower() == 'i':
                return self._press_mario_restart_key()
            
            # Mehrere Methoden versuchen für bessere Kompatibilität
            success = False
            
            # Methode 1: PyAutoGUI press (kurz drücken und loslassen)
            try:
                pyautogui.press(key)
                print(f"✅ PyAutoGUI press '{key}' successful")
                success = True
            except Exception as e1:
                print(f"⚠️ PyAutoGUI press failed: {e1}")
            
            # Methode 2: PyAutoGUI keyDown/keyUp (länger gedrückt halten)
            try:
                pyautogui.keyDown(key)
                time.sleep(0.2)  # Länger halten für bessere Erkennung
                pyautogui.keyUp(key)
                print(f"✅ PyAutoGUI keyDown/keyUp '{key}' successful")
                success = True
            except Exception as e2:
                print(f"⚠️ PyAutoGUI keyDown/keyUp failed: {e2}")
            
            # Methode 3: Keyboard Library
            try:
                keyboard.press_and_release(key)
                print(f"✅ Keyboard library '{key}' successful")
                success = True
            except Exception as e3:
                print(f"⚠️ Keyboard library failed: {e3}")
            
            # Methode 4: Keyboard Library mit hold
            try:
                keyboard.press(key)
                time.sleep(0.2)
                keyboard.release(key)
                print(f"✅ Keyboard library hold '{key}' successful")
                success = True
            except Exception as e4:
                print(f"⚠️ Keyboard library hold failed: {e4}")
                
            if not success:
                print(f"❌ All methods failed for key '{key}'")
            else:
                print(f"🎯 Key '{key}' pressed successfully with at least one method")
                
            # Zusätzliche Wartezeit nach dem Tastendruck
            time.sleep(0.3)
                
        except Exception as e:
            print(f"❌ Critical error pressing key '{key}': {e}")

    def _press_mario_restart_key(self):
        """
        Speziell optimierte Methode für die Mario Restart 'i' Taste
        """
        print("🎮 Attempting Mario restart with 'i' key...")
        success_count = 0

            
        # Methode 1: PyAutoGUI mit verschiedenen Timings
        try:
            pyautogui.keyDown('i')
            time.sleep(0.5)  # Länger halten für Mario
            pyautogui.keyUp('i')
            success_count += 1
            print(f"  ✅ PyAutoGUI attempt {attempt + 1} successful")
        except Exception as e:
            print(f"  ⚠️ PyAutoGUI attempt {attempt + 1} failed: {e}")
        
        time.sleep(0.2)
        
        # Methode 2: Keyboard Library
        try:
            keyboard.press('i')
            time.sleep(0.5)
            keyboard.release('i')
            success_count += 1
            print(f"  ✅ Keyboard library attempt {attempt + 1} successful")
        except Exception as e:
            print(f"  ⚠️ Keyboard library attempt {attempt + 1} failed: {e}")
        
        time.sleep(0.3)
        
        # Methode 3: Kurzer Tastendruck
        try:
            pyautogui.press('i')
            success_count += 1
            print(f"  ✅ Quick press attempt {attempt + 1} successful")
        except Exception as e:
            print(f"  ⚠️ Quick press attempt {attempt + 1} failed: {e}")
    
        if success_count > 0:
            print(f"🎯 Mario restart key pressed successfully ({success_count} methods worked)")
        else:
            print("❌ All Mario restart key methods failed")
        
        # Längere Wartezeit nach Mario restart key
        time.sleep(1.0)
        

class ManualController:
    """
    Klasse für manuelle Spielsteuerung (zum Testen)
    """
    
    def __init__(self, game_controller: GameController):
        self.game_controller = game_controller
        self.is_active = False
    
    def start_manual_control(self):
        """
        Startet die manuelle Steuerung
        """        
        self.is_active = True        
        print("Manuelle Steuerung gestartet. A=Links, D=Rechts, V=Springen, ESC=Beenden.")
        
        while self.is_active:
            try:
                if keyboard.is_pressed('a'):
                    self.game_controller.perform_action(Action.LEFT)
                elif keyboard.is_pressed('d'):
                    self.game_controller.perform_action(Action.RIGHT)
                elif keyboard.is_pressed('v'):
                    self.game_controller.perform_action(Action.JUMP)
                elif keyboard.is_pressed('down'):
                    self.game_controller.perform_action(Action.DOWN)
                elif keyboard.is_pressed('esc'):
                    self.stop_manual_control()
                else:
                    self.game_controller.perform_action(Action.NOTHING)
                
                time.sleep(0.016)  # ~60 FPS
                
            except Exception as e:
                print(f"Fehler bei manueller Steuerung: {e}")
                break
    
    def stop_manual_control(self):
        """
        Stoppt die manuelle Steuerung
        """
        self.is_active = False
        self.game_controller.cleanup()
        print("Manuelle Steuerung gestoppt.")
