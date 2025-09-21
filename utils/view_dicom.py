import pydicom
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from pydicom.pixel_data_handlers.util import apply_voi_lut
import traceback

class DICOMSaver:
    def __init__(self, image_path, image_type="mask"):
        """
        Initialize the DICOM saver and save frames.
        
        Args:
            image_path (str): Path to the DICOM file
            image_type (str): Type of image - "mask" for segmentation masks or "ct" for CT scans
        """
        try:
            # Read the DICOM file with force=True to handle potential issues
            self.dicom_data = pydicom.dcmread(image_path, force=True)
            self.image_type = image_type
            
            # Check if pixel data exists
            if not hasattr(self.dicom_data, 'pixel_array'):
                print(f"No pixel data found in {image_path}. This may be a DICOM file without image data.")
                print("Available attributes:", dir(self.dicom_data))
                return
            
            # Use the function to process the DICOM data
            self.image = self.read_dicom(self.dicom_data)
            
            self.num_slices = self.image.shape[0] if self.image.ndim == 3 else 1
            
            # Get the base name of the DICOM file (without extension) for the output directory
            self.base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Create a directory to save the images
            self.output_dir = os.path.join(os.path.dirname(image_path), f"{self.base_name}_{self.image_type}")
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save each frame as a separate image
            self.save_frames()
        except Exception as e:
            print(f"Error processing DICOM file: {e}")
            print(traceback.format_exc())
            print("\nDICOM file details:")
            try:
                for elem in self.dicom_data:
                    print(f"  {elem.name}: {repr(elem.value)[:100]}")
            except:
                print("  Could not display DICOM elements")

    def read_dicom(self, dicom, voi_lut=True, fix_monochrome=True):
        """
        Process the DICOM data and return a numpy array.
        Different processing for CT scans vs segmentation masks.
        """
        try:
            # Get pixel data
            data = dicom.pixel_array.astype(np.float32)
            
            if self.image_type == "ct":  # For the Pancreas CT dataset
                # For CT images, apply Hounsfield Unit conversion if needed
                if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
                    data = data * dicom.RescaleSlope + dicom.RescaleIntercept
                
                # Apply window/level settings if provided or use defaults for CT
                if voi_lut and hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                    window_center = dicom.WindowCenter
                    window_width = dicom.WindowWidth
                    
                    # If multiple window settings are available, use the first one
                    if isinstance(window_center, pydicom.multival.MultiValue):
                        window_center = window_center[0]
                    if isinstance(window_width, pydicom.multival.MultiValue):
                        window_width = window_width[0]
                    
                    # Apply window/level
                    data = self.apply_window_level(data, window_center, window_width)
                else:
                    # Default windowing for CT if not specified (soft tissue window)
                    data = self.apply_window_level(data, 40, 400)
                    
            else:  # For masks and other images
                # Apply the VOI LUT if it's available in the DICOM metadata
                if voi_lut:
                    try:
                        data = apply_voi_lut(data, dicom)
                    except:
                        print("Warning: Could not apply VOI LUT, using pixel data directly")
                
                # Fix inversion for MONOCHROME1 (if needed)
                if fix_monochrome and hasattr(dicom, 'PhotometricInterpretation') and dicom.PhotometricInterpretation == "MONOCHROME1":
                    data = np.amax(data) - data
                
                # Normalize the data to 0-255 for image visualization
                if np.min(data) != np.max(data):  # Check to avoid division by zero
                    data = data - np.min(data)
                    data = data / np.max(data)
                    data = (data * 255).astype(np.uint8)
            
            return data
        except Exception as e:
            print(f"Error in read_dicom: {e}")
            print(traceback.format_exc())
            raise

    def apply_window_level(self, data, window_center, window_width):
        """
        Apply windowing to the image to enhance visibility of certain tissues.
        
        Args:
            data: Input image data
            window_center: Center of the window (level)
            window_width: Width of the window
            
        Returns:
            Windowed image data normalized to 0-255
        """
        lower_bound = window_center - window_width / 2
        upper_bound = window_center + window_width / 2
        
        # Apply window
        data = np.clip(data, lower_bound, upper_bound)
        
        # Normalize to 0-255 (avoid division by zero)
        if upper_bound > lower_bound:
            data = ((data - lower_bound) / (upper_bound - lower_bound)) * 255.0
        else:
            data = np.zeros_like(data)
            
        return data.astype(np.uint8)

    def save_frames(self):
        """Save each frame of the DICOM file as a separate image."""
        # Determine appropriate colormap
        if self.image_type == "ct":
            colormap = plt.cm.gray  # Standard for CT images
        else:
            colormap = plt.cm.bone  # For segmentation masks
            
        for i in range(self.num_slices):
            frame = self.image[i] if self.num_slices > 1 else self.image
            
            # Save the frame as a PNG image
            output_file = os.path.join(self.output_dir, f"{self.base_name}_slice_{i+1}.png")
            plt.imsave(output_file, frame, cmap=colormap)
            print(f"Saved frame {i+1} to {output_file}")

def main():
    """Main function to call the DICOM saver."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <dicom_file> [mask|ct]")
        return
    
    dicom_file = sys.argv[1]
    image_type = "mask"  # Default
    
    if len(sys.argv) == 3:
        image_type = sys.argv[2].lower()
        if image_type not in ["mask", "ct"]:
            print("Image type must be either 'mask' or 'ct'")
            return
    
    DICOMSaver(dicom_file, image_type)

if __name__ == "__main__":
    main()