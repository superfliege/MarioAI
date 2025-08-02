"""
Test-Script für das Mario AI System
Testet die grundlegenden Funktionalitäten ohne GUI
"""

import time
import sys
import os

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from screen_capture import ScreenCapture
from game_controller import GameController, Action
from ai_agent import DQNAgent
from game_environment import GameEnvironment
from training_manager import TrainingManager


def test_screen_capture():
    """
    Testet das Screen Capture System
    """
    print("=== Screen Capture Test ===")
    
    capture = ScreenCapture()
    
    # Verfügbare Fenster anzeigen
    windows = capture.get_available_windows()
    print(f"Verfügbare Fenster ({len(windows)}):")
    for i, window in enumerate(windows[:10]):  # Zeige nur die ersten 10
        print(f"  {i+1}. {window}")
    
    if not windows:
        print("Keine Fenster gefunden!")
        return False
    
    # Erstes Fenster auswählen (zu Testzwecken)
    test_window = windows[0]
    print(f"\nWähle Fenster für Test: {test_window}")
    
    if capture.select_window(test_window):
        print("✓ Fenster erfolgreich ausgewählt")
        
        # Capture starten
        if capture.start_capture():
            print("✓ Screen Capture gestartet")
            
            # 5 Sekunden warten und Frames testen
            print("Teste Frames für 5 Sekunden...")
            for i in range(50):  # 5 Sekunden bei ~10 FPS
                frame = capture.get_current_frame()
                processed_frame = capture.get_processed_frame()
                
                if frame is not None and processed_frame is not None:
                    print(f"  Frame {i+1}: {frame.shape} -> {processed_frame.shape}")
                else:
                    print(f"  Frame {i+1}: Fehler beim Abrufen")
                
                time.sleep(0.1)
            
            capture.stop_capture()
            print("✓ Screen Capture gestoppt")
            return True
        else:
            print("✗ Fehler beim Starten des Screen Capture")
    else:
        print("✗ Fehler beim Auswählen des Fensters")
    
    return False


def test_game_controller():
    """
    Testet das Game Controller System
    """
    print("\n=== Game Controller Test ===")
    
    controller = GameController()
    
    # Teste alle Aktionen
    actions = controller.get_available_actions()
    print(f"Verfügbare Aktionen: {[action.name for action in actions]}")
    
    print("Teste Aktionen (jeweils 0.5 Sekunden)...")
    
    for action in actions:
        print(f"  Teste {action.name}...")
        controller.perform_action(action)
        time.sleep(0.5)
    
    # Cleanup
    controller.cleanup()
    print("✓ Game Controller Test abgeschlossen")
    return True


def test_ai_agent():
    """
    Testet den AI Agent
    """
    print("\n=== AI Agent Test ===")
    
    agent = DQNAgent(input_channels=4, num_actions=5)
    
    # Teste Aktion-Auswahl
    dummy_state = np.zeros((4, 84, 84))
    
    print("Teste Aktionsauswahl...")
    for i in range(5):
        action = agent.select_action(dummy_state, training=True)
        print(f"  Test {i+1}: Aktion {action}")
    
    # Teste Experience Storage
    print("Teste Experience Storage...")
    for i in range(10):
        state = np.random.random((4, 84, 84)).astype(np.float32)
        action = i % 5
        reward = np.random.random()
        next_state = np.random.random((4, 84, 84)).astype(np.float32)
        done = i == 9
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"  Gespeicherte Erfahrungen: {len(agent.memory)}")
    
    # Teste Training Step
    if len(agent.memory) >= agent.batch_size:
        print("Teste Training Step...")
        loss = agent.train_step()
        print(f"  Training Loss: {loss:.6f}")
    
    print("✓ AI Agent Test abgeschlossen")
    return True


def test_integration():
    """
    Testet die Integration aller Komponenten
    """
    print("\n=== Integration Test ===")
    
    # Screen Capture initialisieren
    capture = ScreenCapture()
    windows = capture.get_available_windows()
    
    if not windows:
        print("✗ Keine Fenster für Integration Test verfügbar")
        return False
    
    # Erstes Fenster auswählen
    test_window = windows[0]
    if not capture.select_window(test_window):
        print("✗ Konnte Fenster nicht auswählen")
        return False
    
    if not capture.start_capture():
        print("✗ Konnte Screen Capture nicht starten")
        return False
    
    print(f"✓ Screen Capture für '{test_window}' gestartet")
    
    try:
        # Training Manager initialisieren
        training_manager = TrainingManager(capture)
        print("✓ Training Manager initialisiert")
        
        # Kurzer Funktionstest
        print("Teste Grundfunktionen für 10 Sekunden...")
        
        # Environment zurücksetzen
        state = training_manager.environment.reset()
        print(f"✓ Environment Reset: State Shape {state.shape}")
        
        # Einige Steps ausführen
        for i in range(20):
            action = training_manager.agent.select_action(state, training=True)
            next_state, reward, done, info = training_manager.environment.step(action)
            
            print(f"  Step {i+1}: Aktion={action}, Reward={reward:.3f}, Done={done}")
            
            if done:
                print("  Episode beendet, Reset...")
                state = training_manager.environment.reset()
            else:
                state = next_state
            
            time.sleep(0.2)
        
        # Cleanup
        training_manager.cleanup()
        print("✓ Integration Test erfolgreich")
        return True
        
    except Exception as e:
        print(f"✗ Integration Test Fehler: {e}")
        return False
    finally:
        capture.stop_capture()


def test_input_transmission():
    """
    Erweiterte Tests für Input-Übertragung mit verschiedenen Methoden
    """
    print("\n=== Erweiterte Input-Übertragung Tests ===")
    
    controller = GameController()
    capture = ScreenCapture()
    
    # Verfügbare Fenster anzeigen
    windows = capture.get_available_windows()
    if not windows:
        print("❌ Keine Fenster verfügbar für Input-Tests!")
        return False
    
    print(f"Verfügbare Fenster für Input-Tests ({len(windows)}):")
    for i, window in enumerate(windows[:5]):
        print(f"  {i+1}. {window[0]} (Handle: {window[1]})")
    
    # Benutzer wählen lassen
    try:
        choice = input(f"\nWählen Sie ein Fenster (1-{min(5, len(windows))}): ")
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(windows):
            selected_window = windows[choice_idx]
            window_name, window_handle = selected_window
            
            print(f"Gewähltes Fenster: {window_name}")
            
            # Window Handle setzen
            controller.set_game_window(window_handle)
            capture.set_target_window_by_handle(window_handle)
            
            print("\n🔸 Starte Input-Methoden Tests...")
            print("WICHTIG: Stellen Sie sicher, dass das gewählte Fenster sichtbar ist!")
            input("Drücken Sie Enter um fortzufahren...")
            
            # Teste verschiedene Input-Methoden
            controller.test_input_methods()
            
            print("\n✅ Input-Übertragung Tests abgeschlossen!")
            return True
            
        else:
            print("❌ Ungültige Auswahl!")
            return False
            
    except (ValueError, KeyboardInterrupt):
        print("❌ Test abgebrochen!")
        return False
    finally:
        controller.cleanup()


def test_death_screen_detection():
    """
    Testet die Death Screen-Erkennung
    """
    print("\n=== Death Screen Detection Test ===")
    
    capture = ScreenCapture()
    controller = GameController()
    
    # Verfügbare Fenster anzeigen
    windows = capture.get_available_windows()
    if not windows:
        print("❌ Keine Fenster verfügbar für Death Screen Test!")
        return False
    
    print(f"Verfügbare Fenster für Death Screen Test ({len(windows)}):")
    for i, window in enumerate(windows[:5]):
        print(f"  {i+1}. {window[0]} (Handle: {window[1]})")
    
    # Benutzer wählen lassen
    try:
        choice = input(f"\nWählen Sie ein Fenster (1-{min(5, len(windows))}): ")
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(windows):
            selected_window = windows[choice_idx]
            window_name, window_handle = selected_window
            
            print(f"Gewähltes Fenster: {window_name}")
            
            # Screen Capture initialisieren
            if capture.select_window(window_name):
                capture.start_capture()
                
                # Game Environment erstellen
                env = GameEnvironment(capture, controller)
                
                print("\n🔸 Starte Death Screen Detection Test...")
                print("WICHTIG: Zeigen Sie dem System verschiedene Bildschirme!")
                input("Drücken Sie Enter um mit dem Test zu beginnen...")
                
                # Death Screen Test ausführen
                detected_count = env.test_death_screen_detection(50)
                
                print(f"\n✅ Death Screen Detection Test abgeschlossen!")
                print(f"📊 Ergebnis: {detected_count}/50 Frames als Death Screen erkannt")
                
                capture.stop_capture()
                return True
            else:
                print("❌ Konnte Fenster nicht auswählen!")
                return False
            
        else:
            print("❌ Ungültige Auswahl!")
            return False
            
    except (ValueError, KeyboardInterrupt):
        print("❌ Test abgebrochen!")
        return False
    finally:
        controller.cleanup()


def main():
    """
    Führt alle Tests aus
    """
    print("Mario AI System Test")
    print("=" * 50)
    
    # Importiere numpy hier um sicherzustellen, dass es verfügbar ist
    global np
    import numpy as np
    
    tests = [
        ("Screen Capture", test_screen_capture),
        ("Game Controller", test_game_controller),
        ("AI Agent", test_ai_agent),
        ("Integration", test_integration),
        ("Input Transmission", test_input_transmission),
        ("Death Screen Detection", test_death_screen_detection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nStarte {test_name} Test...")
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✓ {test_name} Test ERFOLGREICH")
            else:
                print(f"✗ {test_name} Test FEHLGESCHLAGEN")
        except Exception as e:
            print(f"✗ {test_name} Test FEHLER: {e}")
            results[test_name] = False
    
    # Zusammenfassung
    print("\n" + "=" * 50)
    print("TEST ZUSAMMENFASSUNG")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nErgebnis: {passed}/{total} Tests erfolgreich")
    
    if passed == total:
        print("🎉 Alle Tests erfolgreich! Das System ist bereit für den Einsatz.")
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Bitte überprüfen Sie die Konfiguration.")


if __name__ == "__main__":
    main()
