# SoloCell

# SoloCell - Single Cell Detection Tool

SoloCell is a Python-based tool designed to detect and analyze single cells in grid-formatted microscopy images. It uses computer vision and machine learning techniques to identify, validate, and optimize the selection of individual cells.

## Features

- Automated detection of single cells in grid patterns
- AI-based validation of cell detection
- Multiple path optimization algorithms for efficient cell selection
- Interactive ROI (Region of Interest) selection
- Confidence-based filtering of detected cells
- Generation of coordinate lists for detected cells
- Visual output with numbered cell positions

## Prerequisites

- Python 3.x
- Required Python packages:
  - OpenCV (cv2)
  - NumPy
  - Pandas
  - TensorFlow
  - scikit-learn
  - PIL (Python Imaging Library)
  - tqdm
  - matplotlib

## Usage Instructions

1. **Setup**
   - Ensure all required packages are installed
   - Have your high-resolution microscopy image ready
   - Have a trained keras model for cell detection

2. **Running the Program**
   - Run `SoloCell_V7.py`
   - Follow the prompts to:
     1. Select an output folder for results
     2. Select your high-resolution image
     3. Select your trained keras model file

3. **Reference Selection Process**
   Use the interactive window to select reference points:
   - First 7 's' presses: Select 7 adjacent cells in a row/column
   - 8th 's' press: Select a seed point for grid detection
   - 9th and 10th 's' presses: Select two points to define the grid angle (these points must be in the same row or column and should be as far apart as possible to ensure accurate slope calculation)
   - Use 'd' to draw additional reference lines if needed
   - Use mouse wheel to scroll up/down
   - Use left/right mouse buttons to pan
   - Use the zoom slider for detailed selection

4. **Confidence Selection**
   - After processing, a histogram will appear
   - Click on the desired confidence threshold
   - Cells with confidence above this threshold will be selected

## Output Files

The program generates several output files in your selected results folder:
- `log.txt`: Detailed processing log
- `Selected_Coordinates.txt`: List of final cell coordinates
- `Selected_Frames.png`: Original image with marked and numbered cells
- `Histogram.png`: Confidence distribution of detected cells
- Excel files with detailed cell data

## Tips for Best Results

1. Ensure your image has good contrast and clear cell boundaries
2. Select reference points carefully for accurate grid alignment
3. Choose a confidence threshold that balances detection accuracy with coverage
4. For the first 7 points, select cells that are clearly visible and aligned
5. When selecting the angle reference points (9th and 10th clicks), choose well-separated cells

## Error Handling

- If the program fails to detect cells, try:
  - Adjusting the image contrast
  - Selecting different reference cells
  - Choosing a lower confidence threshold
  - Ensuring the image is in focus and cells are clearly visible

## Limitations

- Works best with grid-formatted cell layouts
- Requires clear cell boundaries
- Performance depends on image quality and cell visibility
- Requires a pre-trained keras model for cell detection

## Support

For issues and questions, please contact the development team or raise an issue in the repository.

## Important Note

**PLEASE NOTE**: This program is currently a prototype and may encounter errors during operation. The machine learning model has not been extensively trained for all cell types and different microscopy image formats. Results may vary depending on your specific use case. Users should validate results manually and report any issues to improve the system.
