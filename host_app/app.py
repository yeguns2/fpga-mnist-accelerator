import serial
import time
import struct
import numpy as np
import sys  
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageFilter, ImageOps
import io

def preprocess_mnist_like(pil_img: Image.Image) -> np.ndarray:
    """
    Convert canvas image to MNIST-style 28x28 grayscale format.
    Input: PIL Image (Black background, White strokes)
    Output: (28,28) uint8 numpy array
    """
    img = pil_img.convert("L")

    # 0) Ensure consistent background (Black background, White ink)
    arr0 = np.array(img, dtype=np.uint8)
    if arr0.mean() > 127:  
        img = ImageOps.invert(img)
        arr0 = np.array(img, dtype=np.uint8)

    # 1) Isolate ink pixels
    # Threshold is kept low (25) to preserve thin strokes.
    thr = 25
    mask = arr0 > thr
    if not np.any(mask):
        return np.zeros((28, 28), dtype=np.uint8)

    # Find bounding box
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # 2) Crop to bounding box
    img = img.crop((x0, y0, x1, y1))

    # 3) Morphological closing
    # Fills small gaps/loops (common in '9' or '8') using Max+Min filters.
    img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.MinFilter(3))

    # 4) Resize to 20x20 while preserving aspect ratio
    w, h = img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))

    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # 5) Center image on 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top  = (28 - new_h) // 2
    canvas.paste(img, (left, top))

    # 6) Center of Mass (CoM) alignment
    # Calculates the weighted center and shifts it to (14, 14).
    arr = np.array(canvas, dtype=np.float32)
    s = arr.sum()
    if s > 0:
        yy, xx = np.indices(arr.shape)
        cy = (yy * arr).sum() / s
        cx = (xx * arr).sum() / s

        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))

        arr = np.roll(arr, shift_y, axis=0)
        arr = np.roll(arr, shift_x, axis=1)

        # Clean up wrapping artifacts at edges
        if shift_y > 0: arr[:shift_y, :] = 0
        if shift_y < 0: arr[shift_y:, :] = 0
        if shift_x > 0: arr[:, :shift_x] = 0
        if shift_x < 0: arr[:, shift_x:] = 0

    # 7) Final smoothing
    # Applies slight blur to reduce aliasing from resizing.
    canvas2 = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    canvas2 = canvas2.filter(ImageFilter.GaussianBlur(radius=0.6))

    return np.array(canvas2, dtype=np.uint8)

def softmax(logits):
    """
    Compute softmax values for each set of scores in logits.
    Logits are expected to be raw integer values from hardware.
    """
    # Convert to float for calculation
    z = np.array(logits, dtype=np.float64)
    
    # Subtract max for numerical stability (prevents overflow in exp)
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    
    return exp_z / np.sum(exp_z)

app = Flask(__name__)

# ==========================================
# 1. FPGA UART Configuration
# ==========================================
# Check Device Manager for the correct port (e.g., COM3, COM4)
COMPORT = 'COM4' 
BAUDRATE = 115200
 
try:
    ser = serial.Serial(COMPORT, BAUDRATE, timeout=1)
    print(f"✅ FPGA Connected on {COMPORT}")
except serial.SerialException as e:
    print(f"❌ Error: Could not connect to FPGA on {COMPORT}.")
    print("   Please check connections and ensure no other terminals are using the port.")
    sys.exit(1) # Halt execution immediately if FPGA is missing

def send_image_to_fpga(flattened_image):
    """
    Transmits raw image data to FPGA via UART and parses the response.
    Includes HW cycles, SW cycles, and Logits for probability calculation.
    """
    
    try:
        # 1. Clear UART buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 2. Send Image Data 
        img_bytes = bytes(flattened_image)
        ser.write(img_bytes)

        # 3. Read Response 
        # Packet Structure (60 bytes):
        # Magic(4) + HW_Cycles(4) + HW_Label(4) + HW_Logits(40) + SW_Cycles(4) + SW_Label(4)
        response = ser.read(60)
        
        if len(response) < 60:
            return {"error": f"Timeout. Received only {len(response)} bytes"}

        # 4. Parse Binary Data
        # Format: Little Endian (<)
        # I: Unsigned Int (Magic, Cycles, Label)
        # 10i: 10 Signed Ints (Logits)
        # II: Unsigned Int (SW Cycles, SW Label)
        unpacked = struct.unpack('<III10iII', response)
        
        magic_val = unpacked[0]
        hw_cycles = unpacked[1]
        hw_label  = unpacked[2]
        logits    = unpacked[3:13] # Tuple of 10 integers
        sw_cycles = unpacked[13]
        
        # Verify sync marker
        if magic_val != 0xAABBCCDD:
            print(f"Sync Warning: Magic mismatch (Got 0x{magic_val:X})")

        # 5. Calculate Probability (Softmax)
        probs = softmax(logits)
        confidence = probs[hw_label] * 100.0

        # 6. Calculate Speedup Factor
        # Formula: SW Cycles / HW Cycles
        if hw_cycles > 0:
            speedup = sw_cycles / hw_cycles
        else:
            speedup = 0.0

        # 7. Calculate Execution Time (100MHz)
        hw_time = hw_cycles * (1 / 100000000) 

        return {
            "label": hw_label,
            "hw_cycles": hw_cycles,
            "sw_cycles": sw_cycles,
            "hw_time": hw_time,
            "confidence": confidence,
            "speedup": speedup
        }

    except Exception as e:
        print(f"Comm Error: {e}")
        return {"error": str(e)}


# ==========================================
# 2. Web Server Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive Image
        file = request.files['image']
        img = Image.open(file.stream).convert('L') 

        # 2. Preprocess
        img_array = preprocess_mnist_like(img)

        # Save debug image locally
        debug_img = Image.fromarray(img_array)
        debug_img.save("debug_input.png") 
        print("📸 Debug image saved as 'debug_input.png'")
        
        # Flatten for transmission
        flat_data = img_array.flatten().tolist()

        # 3. Perform Inference
        result = send_image_to_fpga(flat_data)

        # 4. Return Results
        if "error" in result:
            return jsonify({'error': result['error']})
            
        return jsonify({
            'prediction': result['label'],
            'hw_cycles': result['hw_cycles'],
            'sw_cycles': result['sw_cycles'],
            'fpga_time': f"{result['hw_time']:.6f}s",
            'confidence': f"{result['confidence']:.2f}%",
            'speedup': f"{result['speedup']:.2f}x"
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)