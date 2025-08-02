import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import cv2
from PIL import Image, ImageTk
import numpy as np
from screen_capture import ScreenCapture
from training_manager import TrainingManager
from game_controller import ManualController


class MarioAIGUI:
    """
    Hauptfenster für die Mario AI Anwendung
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mario AI - Reinforcement Learning")
        self.root.geometry("1200x800")
          # Komponenten
        self.screen_capture = ScreenCapture()
        self.training_manager = None
        self.manual_controller = None
        
        # GUI State
        self.selected_window = None
        self.is_capturing = False
        self.stats_update_thread = None
        self.stats_running = False
        self.agent_testing = False
        
        # Setup GUI
        self.setup_gui()
        self.update_window_list()
        
        # Cleanup beim Schließen
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """
        Erstellt die GUI-Elemente
        """
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
          # Window Selection Frame
        window_frame = ttk.LabelFrame(main_frame, text="Fenster Auswahl", padding="5")
        window_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        window_frame.columnconfigure(0, weight=1)
        
        # Label für Fensterauswahl
        ttk.Label(window_frame, text="Wählen Sie ein Fenster:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Combobox für Fensterauswahl
        self.window_var = tk.StringVar()
        self.window_combobox = ttk.Combobox(window_frame, textvariable=self.window_var, state="readonly")
        self.window_combobox.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Button Frame für die Buttons
        button_frame = ttk.Frame(window_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(button_frame, text="Aktualisieren", command=self.update_window_list).grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        ttk.Button(button_frame, text="Auswählen", command=self.select_window).grid(row=0, column=1, padx=(5, 0), sticky=tk.W)
        
        # Control Frame
        control_frame = ttk.LabelFrame(main_frame, text="Steuerung", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # Capture Controls
        capture_controls_frame = ttk.Frame(control_frame)
        capture_controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.capture_button = ttk.Button(capture_controls_frame, text="Screen Capture starten", 
                                        command=self.toggle_capture, state=tk.DISABLED)
        self.capture_button.grid(row=0, column=0, padx=(0, 5))
        
        self.manual_button = ttk.Button(capture_controls_frame, text="Manuell spielen", 
                                       command=self.toggle_manual_control, state=tk.DISABLED)
        self.manual_button.grid(row=0, column=1, padx=(0, 5))
        
        # Training Controls
        training_controls_frame = ttk.Frame(control_frame)
        training_controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.training_button = ttk.Button(training_controls_frame, text="Training starten", 
                                         command=self.toggle_training, state=tk.DISABLED)
        self.training_button.grid(row=0, column=0, padx=(0, 5))
        
        ttk.Label(training_controls_frame, text="Episoden:").grid(row=0, column=1, padx=(10, 5))
        
        self.episodes_var = tk.StringVar(value="1000")
        episodes_spinbox = tk.Spinbox(training_controls_frame, from_=10, to=10000, 
                                     textvariable=self.episodes_var, width=10)
        episodes_spinbox.grid(row=0, column=2, padx=(0, 5))
        ttk.Button(training_controls_frame, text="Agent testen", 
                  command=self.test_agent, state=tk.NORMAL).grid(row=0, column=3, padx=(10, 0))
        
        self.test_stop_button = ttk.Button(training_controls_frame, text="Test stoppen", 
                                          command=self.stop_agent_test, state=tk.DISABLED)
        self.test_stop_button.grid(row=0, column=4, padx=(5, 0))
        
        # Display Frame
        display_frame = ttk.LabelFrame(main_frame, text="Live-Ansicht", padding="5")
        display_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Canvas für Video Display
        self.canvas = tk.Canvas(display_frame, width=300, height=300, bg='black')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Stats Frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistiken", padding="5")
        stats_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(1, weight=1)
        
        # Status Labels
        status_frame = ttk.Frame(stats_frame)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Bereit", foreground="green")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Statistics Text Area
        self.stats_text = scrolledtext.ScrolledText(stats_frame, width=40, height=20, 
                                                   font=("Consolas", 9))
        self.stats_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons Frame
        buttons_frame = ttk.Frame(stats_frame)
        buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Button(buttons_frame, text="Stats anzeigen", 
                  command=self.show_training_plots).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(buttons_frame, text="Modell speichern", 
                  command=self.save_model).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(buttons_frame, text="Lade AI Model", 
                  command=self.load_model).grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))
    
    def update_window_list(self):
        """
        Aktualisiert die Liste der verfügbaren Fenster
        """
        try:
            windows = self.screen_capture.get_available_windows()
            self.window_combobox['values'] = windows
            if windows:
                self.window_combobox.current(0)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Aktualisieren der Fensterliste: {e}")
    
    def select_window(self):
        """
        Wählt das ausgewählte Fenster aus
        """
        selected = self.window_var.get()
        if not selected:
            messagebox.showwarning("Warnung", "Bitte wählen Sie ein Fenster aus!")
            return
        
        if self.screen_capture.select_window(selected):
            self.selected_window = selected
            self.capture_button.config(state=tk.NORMAL)
            self.manual_button.config(state=tk.NORMAL)
            self.update_status(f"Fenster ausgewählt: {selected}")
            messagebox.showinfo("Erfolg", f"Fenster '{selected}' erfolgreich ausgewählt!")
        else:
            messagebox.showerror("Fehler", f"Konnte Fenster '{selected}' nicht auswählen!")
    
    def toggle_capture(self):
        """
        Startet/Stoppt die Bildschirmaufnahme
        """
        if not self.is_capturing:
            if self.screen_capture.start_capture():
                self.is_capturing = True
                self.capture_button.config(text="Screen Capture stoppen")
                self.training_button.config(state=tk.NORMAL)
                self.update_status("Screen Capture läuft")
                
                # Training Manager initialisieren
                if not self.training_manager:
                    self.training_manager = TrainingManager(self.screen_capture)
                
                # Video Display starten
                self.start_video_display()
            else:
                messagebox.showerror("Fehler", "Konnte Screen Capture nicht starten!")
        else:
            self.screen_capture.stop_capture()
            self.is_capturing = False
            self.capture_button.config(text="Screen Capture starten")
            self.training_button.config(state=tk.DISABLED)
            self.update_status("Screen Capture gestoppt")
            
            # Video Display stoppen
            self.stop_video_display()
    
    def toggle_manual_control(self):
        """
        Startet/Stoppt die manuelle Steuerung
        """
        if not self.manual_controller:
            if not self.training_manager:
                self.training_manager = TrainingManager(self.screen_capture)
            
            self.manual_controller = ManualController(self.training_manager.game_controller)
            
            # Starte manuelle Steuerung in separatem Thread
            manual_thread = threading.Thread(target=self.manual_controller.start_manual_control)
            manual_thread.daemon = True
            manual_thread.start()
            
            self.manual_button.config(text="Manuell stoppen")
            self.update_status("Manuelle Steuerung aktiv")
        else:
            self.manual_controller.stop_manual_control()
            self.manual_controller = None
            self.manual_button.config(text="Manuell spielen")
            self.update_status("Manuelle Steuerung gestoppt")
    
    def toggle_training(self):
        """
        Startet/Stoppt das Training
        """
        if not self.training_manager.is_training:
            try:
                episodes = int(self.episodes_var.get())
                if self.training_manager.start_training(episodes):
                    self.training_button.config(text="Training stoppen")
                    self.update_status("Training läuft...")
                    self.start_stats_update()
                else:
                    messagebox.showerror("Fehler", "Konnte Training nicht starten!")
            except ValueError:
                messagebox.showerror("Fehler", "Ungültige Anzahl von Episoden!")
        else:
            self.training_manager.stop_training()
            self.training_button.config(text="Training starten")
            self.update_status("Training gestoppt")
            self.stop_stats_update()    
        
    def test_agent(self):
        if self.agent_testing:
            messagebox.showwarning("Warnung", "Agent-Test läuft bereits!")
            return
            
        if not self.training_manager:
            self.training_manager = TrainingManager(self.screen_capture)
        
        self.agent_testing = True
        self.test_stop_button.config(state=tk.NORMAL)
        self.update_status("Agent wird getestet...")
        
        # Training Manager startet den Test in seinem eigenen Thread
        self.training_manager.test_agent(episodes=3)
        
        # Überwache den Test-Status
        self._monitor_test_status()
    
    def stop_agent_test(self):
        """
        Stoppt den Agent-Test
        """
        if self.training_manager and self.agent_testing:
            self.training_manager.stop_agent_test()
            self.agent_testing = False
            self.test_stop_button.config(state=tk.DISABLED)
            self.update_status("Agent-Test gestoppt")
    
    def _monitor_test_status(self):
        """
        Überwacht den Test-Status und aktualisiert die GUI entsprechend
        """
        if self.training_manager and hasattr(self.training_manager, 'is_testing'):
            if self.training_manager.is_testing:
                # Test läuft noch, plane nächste Überprüfung
                self.root.after(1000, self._monitor_test_status)  # Prüfe jede Sekunde
            else:
                # Test ist beendet
                self.agent_testing = False
                self.test_stop_button.config(state=tk.DISABLED)
                self.update_status("Agent-Test abgeschlossen")
        else:
            # Fallback: Test ist beendet
            self.agent_testing = False
            self.test_stop_button.config(state=tk.DISABLED)
            self.update_status("Agent-Test abgeschlossen")
    
    def start_video_display(self):
        """
        Startet die Video-Anzeige
        """
        self.update_video_display()
    
    def stop_video_display(self):
        """
        Stoppt die Video-Anzeige
        """
        pass  # Update läuft automatisch weiter und stoppt wenn kein Frame verfügbar
    
    def update_video_display(self):
        """
        Aktualisiert die Video-Anzeige
        """
        if self.is_capturing:
            # Versuche Frame zu holen
            if self.training_manager and self.training_manager.environment:
                frame = self.training_manager.environment.get_frame_for_display()
            else:
                frame = self.screen_capture.get_current_frame()
            
            if frame is not None:
                # Frame für Display vorbereiten
                display_frame = self._prepare_frame_for_display(frame)
                
                if display_frame is not None:
                    # Canvas aktualisieren
                    self.canvas.delete("all")
                    
                    # Frame in Canvas zentrieren
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:  # Canvas ist initialisiert
                        x = (canvas_width - display_frame.width()) // 2
                        y = (canvas_height - display_frame.height()) // 2
                        self.canvas.create_image(x, y, anchor=tk.NW, image=display_frame)
                        self.canvas.image = display_frame  # Referenz behalten
        
        # Nächstes Update planen
        self.root.after(33, self.update_video_display)  # ~30 FPS
    
    def _prepare_frame_for_display(self, frame):
        """
        Bereitet einen Frame für die Anzeige vor
        """
        try:
            if frame.shape[2] == 1:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 3:  # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Größe anpassen (max 300x300)
            h, w = frame.shape[:2]
            max_size = 300
            
            if max(h, w) > max_size:
                if h > w:
                    new_h, new_w = max_size, int(w * max_size / h)
                else:
                    new_h, new_w = int(h * max_size / w), max_size
                frame = cv2.resize(frame, (new_w, new_h))
            
            # In PIL Image konvertieren
            pil_image = Image.fromarray(frame)
            return ImageTk.PhotoImage(pil_image)
            
        except Exception as e:
            print(f"Fehler bei Frame-Vorbereitung: {e}")
            return None
    
    def start_stats_update(self):
        """
        Startet die Statistik-Updates
        """
        if not self.stats_running:
            self.stats_running = True
            self.stats_update_thread = threading.Thread(target=self._update_stats_loop)
            self.stats_update_thread.daemon = True
            self.stats_update_thread.start()
    
    def stop_stats_update(self):
        """
        Stoppt die Statistik-Updates
        """
        self.stats_running = False
    
    def _update_stats_loop(self):
        """
        Aktualisiert Statistiken in regelmäßigen Abständen
        """
        while self.stats_running:
            try:
                if self.training_manager:
                    stats = self.training_manager.get_training_stats()
                    self._update_stats_display(stats)
                time.sleep(2)  # Update alle 2 Sekunden
            except Exception as e:
                print(f"Fehler bei Stats-Update: {e}")
                time.sleep(5)
    
    def _update_stats_display(self, stats):
        """
        Aktualisiert die Statistik-Anzeige
        """
        def update_ui():
            self.stats_text.delete(1.0, tk.END)
            
            stats_text = f"""TRAINING STATISTIKEN
{'='*30}

Episode: {stats.get('current_episode', 0)}/{stats.get('max_episodes', 0)}
Gesamt Steps: {stats.get('total_steps', 0):,}
Epsilon: {stats.get('epsilon', 0):.4f}

PERFORMANCE (Letzte 100 Episoden)
{'='*30}
Durchschn. Reward: {stats.get('avg_reward_last_100', 0):.2f}
Durchschn. Länge: {stats.get('avg_length_last_100', 0):.1f}
Durchschn. Loss: {stats.get('avg_loss_last_100', 0):.6f}

ZEIT STATISTIKEN
{'='*30}"""
            
            if 'training_duration' in stats:
                duration = stats['training_duration']
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                
                stats_text += f"""
Training Zeit: {hours:02d}:{minutes:02d}:{seconds:02d}
Episoden/Stunde: {stats.get('episodes_per_hour', 0):.1f}"""
            
            stats_text += f"""

STATUS
{'='*30}
Training aktiv: {'Ja' if stats.get('is_training', False) else 'Nein'}
Screen Capture: {'Aktiv' if self.is_capturing else 'Inaktiv'}
Gewähltes Fenster: {self.selected_window or 'Keines'}
"""
            
            self.stats_text.insert(1.0, stats_text)
        
        # UI Update im Main Thread ausführen
        self.root.after(0, update_ui)
    
    def show_training_plots(self):
        """
        Zeigt Trainings-Plots an
        """
        if self.training_manager and self.training_manager.agent:
            try:
                self.training_manager.agent.plot_training_stats()
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Anzeigen der Plots: {e}")
        else:
            messagebox.showwarning("Warnung", "Keine Trainingsdaten verfügbar!")
    def save_model(self):
        """
        Speichert das aktuelle Modell
        """
        if self.training_manager and self.training_manager.agent:
            try:
                self.training_manager._save_model()
                messagebox.showinfo("Erfolg", "Modell erfolgreich gespeichert!")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Speichern: {e}")
        else:
            messagebox.showwarning("Warnung", "Kein Modell zum Speichern verfügbar!")
    
    def load_model(self):
        """
        Lädt das neueste AI Modell
        """
        try:
            # Initialisiere Training Manager falls nötig
            if not self.training_manager:
                self.training_manager = TrainingManager(self.screen_capture)
            
            # Lade das neueste Modell
            success = self.training_manager.load_latest_model()
            
            if success:
                messagebox.showinfo("Erfolg", "AI Modell erfolgreich geladen!")
                self.update_status("AI Modell geladen")
            else:
                messagebox.showwarning("Warnung", "Kein gespeichertes Modell gefunden!")
                
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden des Modells: {e}")
            self.update_status("Fehler beim Laden des Modells")
    
    def update_status(self, status_text):
        """
        Aktualisiert den Status-Text
        """
        def update():
            self.status_label.config(text=status_text)
        
        self.root.after(0, update)
    
    def on_closing(self):
        """
        Wird beim Schließen des Fensters aufgerufen
        """
        try:
            # Training stoppen
            if self.training_manager:
                self.training_manager.cleanup()
            
            # Screen Capture stoppen
            if self.is_capturing:
                self.screen_capture.stop_capture()
            
            # Stats Update stoppen
            self.stop_stats_update()
            
            # Manuelle Steuerung stoppen
            if self.manual_controller:
                self.manual_controller.stop_manual_control()
                
        except Exception as e:
            print(f"Fehler beim Cleanup: {e}")
        
        self.root.destroy()
    
    def run(self):
        """
        Startet die GUI
        """
        self.root.mainloop()


def main():
    """
    Hauptfunktion
    """
    try:
        app = MarioAIGUI()
        app.run()
    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
