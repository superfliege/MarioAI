"""
Spezielles Test-Tool für Death Screen-Erkennung
"""

import time
import numpy as np
from screen_capture import ScreenCapture
from game_controller import GameController
from game_environment import GameEnvironment


def main():
    """
    Interaktiver Death Screen Detection Test
    """
    print("🎮 Death Screen Detection Test Tool")
    print("=" * 50)
    
    capture = ScreenCapture()
    controller = GameController()
    
    # Verfügbare Fenster anzeigen
    windows = capture.get_available_windows()
    if not windows:
        print("❌ Keine Fenster verfügbar!")
        return
    
    print(f"Verfügbare Fenster ({len(windows)}):")
    for i, (window_name, window_handle) in enumerate(windows[:10]):
        print(f"  {i+1}. {window_name}")
    
    # Fensterauswahl
    try:
        choice = int(input(f"\nWählen Sie ein Fenster (1-{min(10, len(windows))}): ")) - 1
        if 0 <= choice < len(windows):
            window_name, window_handle = windows[choice]
            print(f"Ausgewähltes Fenster: {window_name}")
        else:
            print("❌ Ungültige Auswahl!")
            return
    except ValueError:
        print("❌ Ungültige Eingabe!")
        return
    
    # Screen Capture starten
    if not capture.select_window(window_name):
        print("❌ Konnte Fenster nicht auswählen!")
        return
    
    if not capture.start_capture():
        print("❌ Konnte Screen Capture nicht starten!")
        return
    
    # Environment erstellen
    env = GameEnvironment(capture, controller)
    
    print("\n🔍 Death Screen Detection läuft...")
    print("Drücken Sie Ctrl+C zum Beenden")
    print("-" * 50)
    
    try:
        frame_count = 0
        death_screens = 0
        
        while True:
            frame_count += 1
            
            # Aktuellen Frame holen
            current_frame = capture.get_processed_frame()
            if current_frame is not None:
                
                # Death Screen Test
                is_death = env._is_death_screen(current_frame)
                
                # Statistiken berechnen
                black_pixels = np.sum(current_frame < 0.15)
                total_pixels = current_frame.size
                black_percentage = black_pixels / total_pixels
                
                if is_death:
                    death_screens += 1
                    print(f"Frame {frame_count:4d}: 💀 DEATH SCREEN - {black_percentage:.1%} schwarz")
                elif frame_count % 30 == 0:  # Alle 30 Frames normalen Status zeigen
                    print(f"Frame {frame_count:4d}: ✅ Normal - {black_percentage:.1%} schwarz")
                
                # Zusammenfassung alle 300 Frames (ca. 10 Sekunden)
                if frame_count % 300 == 0:
                    detection_rate = (death_screens / frame_count) * 100
                    print(f"\n📊 Zwischenbilanz nach {frame_count} Frames:")
                    print(f"   Death Screens erkannt: {death_screens} ({detection_rate:.1f}%)")
                    print("-" * 50)
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print(f"\n\n📊 Finale Statistiken:")
        print(f"   Frames analysiert: {frame_count}")
        print(f"   Death Screens erkannt: {death_screens}")
        if frame_count > 0:
            detection_rate = (death_screens / frame_count) * 100
            print(f"   Detection Rate: {detection_rate:.1f}%")
        print("\n✅ Test beendet!")
    
    finally:
        capture.stop_capture()
        controller.cleanup()


if __name__ == "__main__":
    main()
