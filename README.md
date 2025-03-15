# Seam Carving Project

This project implements a seam carving algorithm to resize images by removing seams of low energy.

## Features

- Resize images by removing vertical and horizontal seams.
- Visualize energy maps and removed seams.
- Capable of sizing down images by up to 50%.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/engy-58/seam-carving
   cd seam-carving
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the seam carving algorithm with the following command:

```bash
python seam_carver.py C:\path\to\image.jpg --width 300 --height 200 --output examples/output
```

- **`--image_path`**: Path to the input image.
- **`--width`**: Target width for the resized image.
- **`--height`**: Target height for the resized image.
- **`--output`**: Directory where the results will be saved.

## Examples

Input and output images are stored in the `examples/` directory.

## License

This project is licensed under the MIT License.