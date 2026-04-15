import tkinter as tk
from PIL import Image, ImageDraw
import io
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# Projekt-Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models import QuickDrawNN, QuickDrawCNN
import config

# ── Konstanten ────────────────────────────────────────────────────────────────
CLASSES     = config.CLASSES
CANVAS_SIZE = 580
BAR_HEIGHT  = 30
BAR_WIDTH   = 390
PANEL_W     = 520

COLORS = {
    "apple":      "#ff6b6b",
    "candle":     "#ffa94d",
    "eyeglasses": "#51cf66",
    "fork":       "#cc5de8",
    "star":       "#ffd43b",
}

HINTS = {
    "apple":      "Kreis mit Stiel oben",
    "candle":     "Rechteck + Flamme oben",
    "eyeglasses": "Zwei Kreise verbunden",
    "fork":       "3-4 Zinken + langer Griff",
    "star":       "5 Zacken von der Mitte",
}

BG    = "#111111"
PANEL = "#1c1c1c"
FG    = "#eeeeee"

# ── Modelle direkt laden ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Lade Modelle auf: {DEVICE}")

def load_model(model_class, path):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

nn_model  = load_model(QuickDrawNN,  "train_results/nn_model.pth")
cnn_model = load_model(QuickDrawCNN, "train_results/cnn_model.pth")
print("Modelle geladen.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> torch.Tensor:
    image = pil_image.convert("L")

    # Schwellenwert: dunkle Pixel → reines Schwarz
    arr = np.array(image)
    arr[arr < 30] = 0
    image = Image.fromarray(arr)

    # Bounding Box + zentrieren
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        w, h = image.size
        max_side = max(w, h)
        padding  = int(max_side * 0.25)
        new_size = max_side + padding * 2
        padded   = Image.new("L", (new_size, new_size), 0)
        padded.paste(image, ((new_size - w) // 2, (new_size - h) // 2))
        image = padded

    image = image.resize((28, 28), Image.LANCZOS)
    arr   = np.array(image).astype("float32") / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ── Vorhersage ────────────────────────────────────────────────────────────────
def predict(model, tensor) -> dict:
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}


# ══════════════════════════════════════════════════════════════════════════════
class QuickDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickDraw AI — CNN vs. NN")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.last_x    = None
        self.last_y    = None

        self._build_ui()

        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth()  - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 0))
        tk.Label(header, text="AI Drawing Recognition",
                 bg=BG, fg=FG, font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(header, text="CNN vs. NN — zeichne eines der 5 Objekte",
                 bg=BG, fg="#555", font=("Segoe UI", 11)).pack(anchor="w", pady=(2, 10))

        # Legende
        legend = tk.Frame(self.root, bg=BG)
        legend.pack(fill="x", padx=24)
        for cls in CLASSES:
            f = tk.Frame(legend, bg=BG)
            f.pack(side="left", padx=8)
            tk.Label(f, text="●", bg=BG, fg=COLORS[cls],
                     font=("Segoe UI", 13)).pack(side="left")
            lbl = tk.Label(f, text=cls.capitalize(), bg=BG, fg="#999",
                           font=("Segoe UI", 11), cursor="hand2")
            lbl.pack(side="left", padx=(2, 0))
            lbl.bind("<Enter>", lambda e, c=cls: self.hint_var.set(
                f"Tipp  {c.capitalize()}: {HINTS[c]}"))
            lbl.bind("<Leave>", lambda e: self.hint_var.set(
                "← Hover über eine Klasse für Zeichentipps"))

        # Hint Box
        self.hint_var = tk.StringVar(value="← Hover über eine Klasse für Zeichentipps")
        tk.Label(self.root, textvariable=self.hint_var,
                 bg="#181818", fg="#555", font=("Segoe UI", 10),
                 anchor="w", padx=14, pady=7
                 ).pack(fill="x", padx=24, pady=(8, 10))

        # Hauptbereich
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # Linke Spalte: Canvas
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", anchor="n")

        tk.Label(left, text="Drawing Area", bg=BG, fg="#3498db",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="#0a0a0a", cursor="crosshair",
            highlightthickness=2, highlightbackground="#2a2a2a"
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._start_draw)
        self.canvas.bind("<B1-Motion>",       self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(12, 0))
        tk.Button(btn_row, text="Recognize",
                  bg="#ffffff", fg="#111", relief="flat",
                  font=("Segoe UI", 13, "bold"), cursor="hand2",
                  command=self._recognize
                  ).pack(side="left", fill="x", expand=True, ipady=12, padx=(0, 8))
        tk.Button(btn_row, text="Clear",
                  bg="#1e1e1e", fg="#888", relief="flat",
                  font=("Segoe UI", 13), cursor="hand2",
                  command=self._clear
                  ).pack(side="left", fill="x", expand=True, ipady=12)

        # Rechte Spalte: Ergebnisse
        right = tk.Frame(main, bg=BG, width=PANEL_W)
        right.pack(side="left", anchor="n", padx=(28, 0), fill="both", expand=True)
        right.pack_propagate(False)

        tk.Label(right, text="Recognition Results", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))

        self.cnn_panel = self._build_panel(right, "CNN", "#51cf66")
        tk.Frame(right, bg="#2a2a2a", height=2).pack(fill="x", pady=14)
        self.nn_panel  = self._build_panel(right, "NN",  "#339af0")

        self.status_var = tk.StringVar(value="Zeichne etwas und drücke Recognize.")
        tk.Label(right, textvariable=self.status_var,
                 bg=BG, fg="#555", font=("Segoe UI", 10),
                 wraplength=PANEL_W - 20, justify="left", anchor="w"
                 ).pack(anchor="w", pady=(14, 0))

    def _build_panel(self, parent, title, accent):
        frame = tk.Frame(parent, bg=PANEL, padx=16, pady=14)
        frame.pack(fill="x")
        tk.Frame(frame, bg=accent, height=4).pack(fill="x", pady=(0, 10))

        hdr = tk.Frame(frame, bg=PANEL)
        hdr.pack(fill="x", pady=(0, 10))
        tk.Label(hdr, text=title, bg=PANEL, fg=accent,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        top_lbl = tk.Label(hdr, text="—", bg=PANEL, fg="#888",
                           font=("Segoe UI", 12))
        top_lbl.pack(side="right")

        bars = {}
        for cls in CLASSES:
            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=cls.capitalize(), bg=PANEL, fg="#888",
                     font=("Segoe UI", 11), width=11, anchor="w").pack(side="left")
            track = tk.Canvas(row, bg="#222", height=BAR_HEIGHT,
                              width=BAR_WIDTH, highlightthickness=0)
            track.pack(side="left", padx=(4, 10))
            bar_id = track.create_rectangle(0, 0, 0, BAR_HEIGHT,
                                            fill=COLORS[cls], outline="")
            pct_lbl = tk.Label(row, text="0.0%", bg=PANEL, fg="#555",
                               font=("Segoe UI", 11, "bold"), width=6, anchor="e")
            pct_lbl.pack(side="left")
            bars[cls] = (track, bar_id, pct_lbl)

        return {"bars": bars, "top_lbl": top_lbl, "accent": accent}

    # ── Zeichnen ──────────────────────────────────────────────────────────────
    def _start_draw(self, e):
        self.last_x, self.last_y = e.x, e.y

    def _draw(self, e):
        if self.last_x is None:
            return
        self.canvas.create_line(
            self.last_x, self.last_y, e.x, e.y,
            fill="white", width=14, capstyle="round", joinstyle="round"
        )
        self.pil_draw.line(
            [(self.last_x, self.last_y), (e.x, e.y)],
            fill=(255, 255, 255), width=14
        )
        self.last_x, self.last_y = e.x, e.y

    def _stop_draw(self, _):
        self.last_x = self.last_y = None

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self._reset_panel(self.cnn_panel)
        self._reset_panel(self.nn_panel)
        self.status_var.set("Zeichne etwas und drücke Recognize.")

    # ── Predict direkt ────────────────────────────────────────────────────────
    def _recognize(self):
        self.status_var.set("Analysiere…")
        self.root.update()

        try:
            tensor    = preprocess(self.pil_image)
            cnn_preds = predict(cnn_model, tensor)
            nn_preds  = predict(nn_model,  tensor)

            self._update_panel(self.cnn_panel, cnn_preds)
            self._update_panel(self.nn_panel,  nn_preds)

            cnn_top = max(cnn_preds, key=cnn_preds.get)
            nn_top  = max(nn_preds,  key=nn_preds.get)

            if cnn_top != nn_top:
                self.status_var.set(
                    f"⚡ Modelle uneinig:\n"
                    f"CNN → {cnn_top}    NN → {nn_top}\n"
                    f"CNN erkennt lokale Muster, NN globale Pixelverteilung."
                )
            else:
                self.status_var.set(f"✓ Beide Modelle einig: {cnn_top.upper()}")

        except Exception as ex:
            self.status_var.set(f"Fehler: {ex}")

    def _update_panel(self, panel, predictions):
        top_cls  = max(predictions, key=predictions.get)
        top_prob = predictions[top_cls] * 100
        warn     = "  ⚠" if top_prob < 70 else ""
        panel["top_lbl"].config(
            text=f"{top_cls.upper()}  {top_prob:.1f}%{warn}",
            fg=panel["accent"] if top_prob >= 70 else "#ffa94d"
        )
        for cls, (track, bar_id, pct_lbl) in panel["bars"].items():
            pct = predictions.get(cls, 0) * 100
            w   = int(BAR_WIDTH * pct / 100)
            track.coords(bar_id, 0, 0, w, BAR_HEIGHT)
            track.itemconfig(bar_id,
                             fill=COLORS[cls] if cls == top_cls else "#333333")
            pct_lbl.config(
                text=f"{pct:.1f}%",
                fg=COLORS[cls] if cls == top_cls else "#555"
            )

    def _reset_panel(self, panel):
        panel["top_lbl"].config(text="—", fg="#888")
        for cls, (track, bar_id, pct_lbl) in panel["bars"].items():
            track.coords(bar_id, 0, 0, 0, BAR_HEIGHT)
            track.itemconfig(bar_id, fill=COLORS[cls])
            pct_lbl.config(text="0.0%", fg="#555")


if __name__ == "__main__":
    root = tk.Tk()
    QuickDrawApp(root)
    root.mainloop()