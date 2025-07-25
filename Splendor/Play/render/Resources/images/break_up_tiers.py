from PIL import Image
import os

# Define the path to the image
image_path = r"C:\Users\Public\Documents\Python_Files\Splendor\meta\images\tier3.jpg"
output_dir = r"C:\Users\Public\Documents\Python_Files\Splendor\meta\images\sections"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the size of each section
section_width = 712
section_height = 1004

def save_image_sections(image_path, output_dir, section_width, section_height):
    # Open the image
    image = Image.open(image_path)
    image_width, image_height = image.size

    section_index = 0

    for top in range(0, image_height, section_height):
        for left in range(0, image_width, section_width):
            # Calculate the right and bottom coordinates of the section
            right = min(left + section_width, image_width)
            bottom = min(top + section_height, image_height)

            # Crop the section
            section = image.crop((left, top, right, bottom))

            # Define the path for the section image
            section_path = os.path.join(output_dir, f"section_{section_index}.jpg")

            # Save the section image
            section.save(section_path)
            print(f"Saved {section_path}")

            section_index += 1

# Run the function to save image sections
save_image_sections(image_path, output_dir, section_width, section_height)
