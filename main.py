import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.original_image = None
        self.processed_image = None

        # Buttons
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image, bg="lightblue")
        self.btn_sobel = tk.Button(root, text="Apply Sobel", command=self.apply_sobel, bg="lightgreen")
        self.btn_canny = tk.Button(root, text="Apply Canny", command=self.apply_canny, bg="lightcoral")
        self.btn_kmeans = tk.Button(root, text="Apply K-Means", command=self.apply_kmeans, bg="lightyellow")

        # Canvas to display images
        self.canvas_original = tk.Canvas(root, width=400, height=400)
        self.canvas_processed = tk.Canvas(root, width=400, height=400)

        # Thresholding Parameters
        self.threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold",
                                         command=self.apply_threshold)
        
        # K-Means Clustering Parameters
        self.cluster_label = tk.Label(root, text="Number of Clusters:")
        self.cluster_entry = tk.Entry(root)
        self.cluster_entry.insert(0, "3")  # Default number of clusters

        # Layout
        self.canvas_original.pack(side=tk.LEFT)
        self.canvas_processed.pack(side=tk.RIGHT)
        self.threshold_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.cluster_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.cluster_entry.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.btn_load.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.btn_sobel.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.btn_canny.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.btn_kmeans.pack(side=tk.BOTTOM, padx=5, pady=5)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            try:
                self.original_image = cv2.imread(path)
                self.display_image(self.original_image, self.canvas_original)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def display_image(self, image, canvas):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        aspect_ratio = w / h
        # Resize image to fit canvas while maintaining aspect ratio
        if w > h:
            new_w = 400
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = 400
            new_w = int(new_h * aspect_ratio)
        image = cv2.resize(image, (new_w, new_h))
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def apply_sobel(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            self.processed_image = magnitude.astype(np.uint8)
            self.display_image(self.processed_image, self.canvas_processed)
        else:
            messagebox.showerror("Error", "Please load an image first.")

    def apply_canny(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = edges
            self.display_image(self.processed_image, self.canvas_processed)
        else:
            messagebox.showerror("Error", "Please load an image first.")

    def apply_threshold(self, value):
        if self.original_image is not None:
            threshold_value = int(value)
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.processed_image = thresholded_image
            self.display_image(self.processed_image, self.canvas_processed)
        else:
            messagebox.showerror("Error", "Please load an image first.")

    def apply_kmeans(self):
        if self.original_image is not None:
            try:
                num_clusters = int(self.cluster_entry.get())
                if num_clusters <= 0:
                    raise ValueError("Number of clusters must be positive.")
                flattened_image = self.original_image.reshape((-1, 3))
                flattened_image = np.float32(flattened_image)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(flattened_image, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_image = centers[labels.flatten()]
                segmented_image = segmented_image.reshape(self.original_image.shape)
                self.processed_image = segmented_image
                self.display_image(self.processed_image, self.canvas_processed)
            except ValueError as ve:
                messagebox.showerror("Error", str(ve))
        else:
            messagebox.showerror("Error", "Please load an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
