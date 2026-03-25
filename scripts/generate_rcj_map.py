import cv2
import numpy as np

def generate_map():
    # 1 cm = 1 pixel. Line thickness = 2 cm (2 pixels)
    # Total inner field is 158 x 219. 
    # We make a 200 x 260 canvas to give a small border.
    img = np.ones((260, 200), dtype=np.uint8) * 255
    cx, cy = 100, 130 # Center of the field

    # --- Draw Outer White Boundary ---
    # Width: 158 (79 left/right), Height: 219 (109.5 up/down, let's use 109)
    cv2.rectangle(img, (cx - 79, cy - 109), (cx + 79, cy + 109), 0, 2)
    
    # --- Draw Center Line ---
    cv2.line(img, (cx - 79, cy), (cx + 79, cy), 0, 2)
    
    # --- Draw Center Circle ---
    # Diameter 60 -> Radius 30
    cv2.circle(img, (cx, cy), 30, 0, 2)
    
    # --- Draw Top Penalty Area ---
    # Straight line (60 wide, 25 deep from top line)
    cv2.line(img, (cx - 30, cy - 109 + 25), (cx + 30, cy - 109 + 25), 0, 2)
    # Left Arc (Center is X=-30, Y=top+10. Radius 15. Angle 90 to 180)
    cv2.ellipse(img, (cx - 30, cy - 109 + 10), (15, 15), 0, 90, 180, 0, 2)
    # Right Arc (Angle 0 to 90)
    cv2.ellipse(img, (cx + 30, cy - 109 + 10), (15, 15), 0, 0, 90, 0, 2)
    # Vertical side lines (10 long to connect arc to backline)
    cv2.line(img, (cx - 45, cy - 109), (cx - 45, cy - 109 + 10), 0, 2)
    cv2.line(img, (cx + 45, cy - 109), (cx + 45, cy - 109 + 10), 0, 2)

    # --- Draw Bottom Penalty Area ---
    # Straight line (60 wide, 25 deep from bottom line)
    cv2.line(img, (cx - 30, cy + 109 - 25), (cx + 30, cy + 109 - 25), 0, 2)
    # Left Arc (Angle 180 to 270)
    cv2.ellipse(img, (cx - 30, cy + 109 - 10), (15, 15), 0, 180, 270, 0, 2)
    # Right Arc (Angle 270 to 360)
    cv2.ellipse(img, (cx + 30, cy + 109 - 10), (15, 15), 0, 270, 360, 0, 2)
    # Vertical side lines
    cv2.line(img, (cx - 45, cy + 109), (cx - 45, cy + 109 - 10), 0, 2)
    cv2.line(img, (cx + 45, cy + 109), (cx + 45, cy + 109 - 10), 0, 2)

    # --- Draw Dots (Points) ---
    # Located at +/- 40 X, and +/- 45 Y from center
    dot_offsets =[(40, 45), (-40, 45), (40, -45), (-40, -45)]
    for px, py in dot_offsets:
        cv2.circle(img, (cx + px, cy + py), 2, 0, -1) # Draw a filled circle as a dot

    # Save the image
    cv2.imwrite("rcj_map.png", img)
    print("rcj_map.png created.")

    # Generate the ROS YAML config file
    yaml_content = f"""image: rcj_map.png
resolution: 0.01
origin:[-1.0, -1.3, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open("rcj_map.yaml", "w") as f:
        f.write(yaml_content)
    print("rcj_map.yaml created.")

if __name__ == "__main__":
    generate_map()