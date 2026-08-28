import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from global_histogram import global_histogram_equalization
from local_histogram import local_histogram_equalization
from adaptive_histogram import adaptive_histogram_equalization


class HistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Histogram Equalization Lab")
        self.root.geometry("1000x750")

        self.gray_img = None
        self.image_path = None

        # Top control bar
        controls = ttk.Frame(root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls, text="Browse Image", command=self.load_image).pack(side=tk.LEFT, padx=5)

        self.method_var = tk.StringVar(value="Global")
        method_menu = ttk.Combobox(
            controls, textvariable=self.method_var,
            values=["Global", "Local", "Adaptive"], state="readonly", width=12
        )
        method_menu.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls, text="Run", command=self.run_equalization).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=15)

        #  Matplotlib figure embedded in the window 
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        self.image_path = path
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.status_var.set(f"Loaded: {path.split('/')[-1]}")
        self.draw(self.gray_img, self.gray_img, "Original")

    def run_equalization(self):
        if self.gray_img is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        method = self.method_var.get()
        self.status_var.set(f"Running {method} equalization...")
        self.root.update_idletasks()

        if method == "Global":
            result = global_histogram_equalization(self.gray_img)
        elif method == "Local":
            result = local_histogram_equalization(self.gray_img, block_size=32)
        else:
            result = adaptive_histogram_equalization(self.gray_img, tiles=8, clip_limit_factor=2.0)

        self.result = result
        self.draw(self.gray_img, result, method)
        self.status_var.set(f"{method} equalization complete.")

    def draw(self, original, processed, title):
        self.fig.clear()
        axes = self.fig.subplots(2, 2)

        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(processed, cmap="gray")
        axes[0, 1].set_title(f"{title} Equalized")
        axes[0, 1].axis("off")

        hist_orig, _ = np.histogram(original.ravel(), bins=256, range=(0, 256))
        hist_eq, _ = np.histogram(processed.ravel(), bins=256, range=(0, 256))

        axes[1, 0].plot(hist_orig, color="black")
        axes[1, 0].set_title("Original Histogram")
        axes[1, 0].set_xlabel("Intensity")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].plot(hist_eq, color="black")
        axes[1, 1].set_title(f"{title} Equalized Histogram")
        axes[1, 1].set_xlabel("Intensity")
        axes[1, 1].set_ylabel("Frequency")

        self.fig.tight_layout()
        self.canvas.draw()

    def save_output(self):
        if not hasattr(self, "result"):
            messagebox.showwarning("Nothing to save", "Run an equalization method first.")
            return
        self.fig.savefig("gui_output.png", dpi=150)
        messagebox.showinfo("Saved", "Saved comparison as gui_output.png")


if __name__ == "__main__":
    root = tk.Tk()
    app = HistogramApp(root)
    root.mainloop()
