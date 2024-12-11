import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def render_video_from_predictions(predictions, original_images, output_dir='output/saved_video', output_file='segmentation_video.mp4', fps=10):
    """
    Renders a video showing the segmentation overlaid on the original CT slices.

    Args:
    - predictions: A numpy array of shape (n_slices, height, width, n_classes) containing the predicted segmentation masks.
    - original_images: A numpy array of shape (n_slices, height, width) containing the original CT slices.
    - output_dir: Directory to save the output video.
    - output_file: Name of the output video file.
    - fps: Frames per second for the output video.
    """

    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' is ready.")

        # Full path to the output video file
        output_path = os.path.join(output_dir, output_file)

        # Get height and width of the images
        height, width = original_images.shape[1:3]

        # Prepare video writer with a supported codec (e.g., 'XVID', 'H264', or 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try different codecs if 'mp4v' doesn't work
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(original_images.shape[0]):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(original_images[i], cmap='gray', alpha=1)
            ax.imshow(np.argmax(predictions[i], axis=-1), cmap='jet', alpha=0.5)
            ax.axis('off')

            # Save the plot to an image buffer
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Check if the frame size is correct
            if img.shape[1] != width or img.shape[0] != height:
                print(f"Frame size mismatch: got {img.shape[:2]}, expected {(height, width)}. Resizing.")
                img = cv2.resize(img, (width, height))

            # Write the image to the video file
            out.write(img)
            plt.close(fig)

        out.release()
        print(f"Video saved as {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        print("Check if the directory path is correct or if there are any permission issues.")