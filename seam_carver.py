import numpy as np
import cv2
from matplotlib import pyplot as plt

class SeamCarver:
    def __init__(self, image_path):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert from BGR to RGB for display
        self.image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        
    def calculate_energy(self, img):
        """Calculate energy using custom gradient calculation instead of Sobel."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        energy = np.zeros_like(gray, dtype=np.float64)
        
        # Calculate x and y gradients manually
        # For each pixel (except edges), calculate differences with neighbors
        for y in range(1, gray.shape[0]-1):
            for x in range(1, gray.shape[1]-1):
                # x-gradient: difference between right and left pixels
                dx = abs(int(gray[y, x+1]) - int(gray[y, x-1]))
                # y-gradient: difference between bottom and top pixels
                dy = abs(int(gray[y+1, x]) - int(gray[y-1, x]))
                energy[y, x] = dx + dy
        
        # Handle edges by copying adjacent values
        energy[0, :] = energy[1, :]  # Top edge
        energy[-1, :] = energy[-2, :]  # Bottom edge
        energy[:, 0] = energy[:, 1]  # Left edge
        energy[:, -1] = energy[:, -2]  # Right edge
        
        return energy
    
    def visualize_energy(self, image=None):
        """Visualize energy map of the image."""
        if image is None:
            image = self.image
            
        energy = self.calculate_energy(image)
        
        # Normalize energy for better visualization
        energy_min = energy.min()
        energy_max = energy.max()
        energy_normalized = 255 * (energy - energy_min) / (energy_max - energy_min)
        energy_normalized = energy_normalized.astype(np.uint8)
        
        # Apply a color map for better visualization
        energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
        energy_colored = cv2.cvtColor(energy_colored, cv2.COLOR_BGR2RGB)
        
        return energy_colored, energy
    
    def cumulative_energy_map(self, energy):
        """Calculate the cumulative energy map for vertical seam identification."""
        height, width = energy.shape
        cum_energy = energy.copy()
        
        for i in range(1, height):
            for j in range(width):
                if j == 0:
                    cum_energy[i, j] += min(cum_energy[i-1, j], cum_energy[i-1, j+1])
                elif j == width-1:
                    cum_energy[i, j] += min(cum_energy[i-1, j-1], cum_energy[i-1, j])
                else:
                    cum_energy[i, j] += min(cum_energy[i-1, j-1], cum_energy[i-1, j], cum_energy[i-1, j+1])
        
        return cum_energy
    
    def find_vertical_seam(self, cum_energy):
        """Find the minimum energy vertical seam."""
        height, width = cum_energy.shape
        seam = np.zeros(height, dtype=np.int32)
        
        # Find the minimum value in the last row
        seam[-1] = np.argmin(cum_energy[-1])
        
        # Backtrack to find the path
        for i in range(height-2, -1, -1):
            j = seam[i+1]
            
            if j == 0:
                cols = [0, 1]
            elif j == width-1:
                cols = [width-2, width-1]
            else:
                cols = [j-1, j, j+1]
            
            seam[i] = cols[np.argmin([cum_energy[i, col] for col in cols])]
                
        return seam
    
    def remove_vertical_seam(self, image, seam):
        """Remove a vertical seam from the image."""
        height, width, channels = image.shape
        output = np.zeros((height, width-1, channels), dtype=np.uint8)
        
        for i in range(height):
            col = seam[i]
            for c in range(channels):
                output[i, :, c] = np.delete(image[i, :, c], col)
                
        return output
    
    def visualize_seam(self, image, seam, direction='vertical'):
        """Visualize a single seam on the image."""
        vis_image = image.copy()
        
        if direction == 'vertical':
            for i in range(len(seam)):
                vis_image[i, seam[i]] = [255, 0, 0]  # Red color for the seam
        else:  # horizontal
            for j in range(len(seam)):
                vis_image[seam[j], j] = [255, 0, 0]  # Red color for the seam
        
        return vis_image
    
    def visualize_seams(self, image, all_seams, direction='vertical'):
        """Create a visualization with all removed seams marked."""
        vis_image = image.copy()
        
        if direction == 'vertical':
            for seam in all_seams:
                for i in range(len(seam)):
                    if i < vis_image.shape[0] and seam[i] < vis_image.shape[1]:
                        vis_image[i, seam[i]] = [255, 0, 0]  # Red color for the seam
        else:  # horizontal
            for seam in all_seams:
                for j in range(len(seam)):
                    if seam[j] < vis_image.shape[0] and j < vis_image.shape[1]:
                        vis_image[seam[j], j] = [255, 0, 0]  # Red color for the seam
        
        return vis_image
    
    def carve_vertical(self, target_width):
        """Reduce width to target_width by removing vertical seams."""
        working_image = self.image.copy()
        original_image = self.image.copy()  # Keep original for visualization
        all_seams = []
        
        while working_image.shape[1] > target_width:
            energy = self.calculate_energy(working_image)
            cum_energy = self.cumulative_energy_map(energy)
            seam = self.find_vertical_seam(cum_energy)
            
            # Store the seam coordinates in original image space
            if all_seams:
                # Adjust seam to account for previous removals
                adjusted_seam = self.adjust_seam_coordinates(seam, all_seams, 'vertical')
                all_seams.append(adjusted_seam)
            else:
                all_seams.append(seam.copy())
                
            working_image = self.remove_vertical_seam(working_image, seam)
        
        # Create visualization with all removed seams
        seam_visualization = self.visualize_seams(original_image, all_seams, 'vertical')
        
        return working_image, seam_visualization
    
    def adjust_seam_coordinates(self, new_seam, previous_seams, direction='vertical'):
        """Adjust seam coordinates to account for previously removed seams."""
        adjusted_seam = new_seam.copy()
        
        if direction == 'vertical':
            # For each pixel in the current seam
            for i in range(len(new_seam)):
                # Count how many seams were removed to the left of this position
                offset = 0
                for prev_seam in previous_seams:
                    if i < len(prev_seam) and prev_seam[i] <= adjusted_seam[i] + offset:
                        offset += 1
                adjusted_seam[i] += offset
        else:  # horizontal
            # For each pixel in the current seam
            for j in range(len(new_seam)):
                # Count how many seams were removed above this position
                offset = 0
                for prev_seam in previous_seams:
                    if j < len(prev_seam) and prev_seam[j] <= adjusted_seam[j] + offset:
                        offset += 1
                adjusted_seam[j] += offset
                
        return adjusted_seam
    
    def find_horizontal_seam(self, energy):
        """Find horizontal seam by transposing the energy matrix."""
        transposed_energy = energy.T
        cum_energy = self.cumulative_energy_map(transposed_energy)
        seam = self.find_vertical_seam(cum_energy)
        return seam
    
    def remove_horizontal_seam(self, image, seam):
        """Remove a horizontal seam from the image."""
        height, width, channels = image.shape
        output = np.zeros((height-1, width, channels), dtype=np.uint8)
        
        for j in range(width):
            row = seam[j]
            for c in range(channels):
                output[:, j, c] = np.delete(image[:, j, c], row)
                
        return output
    
    def carve_horizontal(self, target_height):
        """Reduce height to target_height by removing horizontal seams."""
        working_image = self.image.copy()
        original_image = self.image.copy()  # Keep original for visualization
        all_seams = []
        
        while working_image.shape[0] > target_height:
            energy = self.calculate_energy(working_image)
            seam = self.find_horizontal_seam(energy)
            
            # Store the seam coordinates in original image space
            if all_seams:
                # Adjust seam to account for previous removals
                adjusted_seam = self.adjust_seam_coordinates(seam, all_seams, 'horizontal')
                all_seams.append(adjusted_seam)
            else:
                all_seams.append(seam.copy())
                
            working_image = self.remove_horizontal_seam(working_image, seam)
        
        # Create visualization with all removed seams
        seam_visualization = self.visualize_seams(original_image, all_seams, 'horizontal')
        
        return working_image, seam_visualization
    
    def resize(self, new_width=None, new_height=None):
        """Resize the image to the specified dimensions."""
        if new_width is None and new_height is None:
            return self.image.copy(), None, None
        
        resized_image = self.image.copy()
        v_seam_vis = None
        h_seam_vis = None
        
        # Handle vertical resizing (reducing width)
        if new_width is not None and new_width < self.width:
            resized_image, v_seam_vis = self.carve_vertical(new_width)
        
        # Handle horizontal resizing (reducing height)
        if new_height is not None and new_height < self.height:
            # If we've already resized vertically, use that image for horizontal carving
            temp_carver = SeamCarver.__new__(SeamCarver)
            temp_carver.image = resized_image
            temp_carver.height, temp_carver.width = resized_image.shape[:2]
            temp_carver.original = self.original
            resized_image, h_seam_vis = temp_carver.carve_horizontal(new_height)
        
        return resized_image, v_seam_vis, h_seam_vis

def main():
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Seam Carving Algorithm')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--width', type=int, help='Target width')
    parser.add_argument('--height', type=int, help='Target height')
    parser.add_argument('--output', default='result', help='Output directory prefix')
    args = parser.parse_args()
    
    # Extract the image name without extension
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Create output directory for this image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"{image_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    carver = SeamCarver(args.image_path)
    original_height, original_width = carver.height, carver.width
    
    # Set target dimensions (default to half if not specified)
    target_width = args.width if args.width is not None else original_width // 2
    target_height = args.height if args.height is not None else original_height // 2
    
    # Check that dimensions are valid
    if target_width > original_width or target_height > original_height:
        print("Target dimensions must be smaller than original dimensions")
        return
    if target_width < 1 or target_height < 1:
        print("Target dimensions must be positive")
        return
    
    print(f"Resizing image from {original_width}x{original_height} to {target_width}x{target_height}...")
    
    # Generate energy map visualization
    energy_colored, _ = carver.visualize_energy()
    
    # Perform seam carving
    resized_image, v_seam_vis, h_seam_vis = carver.resize(target_width, target_height)
    
    # Plot results in an organized way
    plt.figure(figsize=(16, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(carver.image)
    plt.title(f'Original ({original_width}x{original_height})', fontsize=12)
    plt.axis('off')
    
    # Energy map
    plt.subplot(2, 3, 2)
    plt.imshow(energy_colored)
    plt.title('Energy Map', fontsize=12)
    plt.axis('off')
    
    # Resized image
    plt.subplot(2, 3, 3)
    plt.imshow(resized_image)
    plt.title(f'Resized ({target_width}x{target_height})', fontsize=12)
    plt.axis('off')
    
    # Vertical seam visualization
    if v_seam_vis is not None:
        plt.subplot(2, 3, 5)
        plt.imshow(v_seam_vis)
        plt.title('Vertical Seams Removed', fontsize=12)
        plt.axis('off')
    
    # Horizontal seam visualization
    if h_seam_vis is not None:
        plt.subplot(2, 3, 6)
        plt.imshow(h_seam_vis)
        plt.title('Horizontal Seams Removed', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # Save individual images
    cv2.imwrite(os.path.join(output_dir, "original.png"), cv2.cvtColor(carver.image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "energy_map.png"), cv2.cvtColor(energy_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "resized.png"), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    
    if v_seam_vis is not None:
        cv2.imwrite(os.path.join(output_dir, "vertical_seams.png"), cv2.cvtColor(v_seam_vis, cv2.COLOR_RGB2BGR))
    
    if h_seam_vis is not None:
        cv2.imwrite(os.path.join(output_dir, "horizontal_seams.png"), cv2.cvtColor(h_seam_vis, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved to {output_dir}")
    print(f"Comparison image saved as {comparison_path}")

if __name__ == "__main__":
    main()