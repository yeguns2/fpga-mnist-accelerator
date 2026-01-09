import serial
import time
import struct
import numpy as np
import sys
import os
import torch

# -----------------------------------------------------------------------------
# This file is intended as a DEBUG utility
# -----------------------------------------------------------------------------
from mnist_train import SmallCNN

# =============================================================================
# [CONFIG] Update these for your environment
# =============================================================================
SERIAL_PORT = 'COM4'    # UART COM port (check Device Manager on Windows)
BAUD_RATE   = 115200    # UART baud rate

# =============================================================================
# UART Resync Helpers
# =============================================================================
MAGIC = 0xAABBCCDD
MAGIC_BYTES = struct.pack('<I', MAGIC)

def read_exact(ser, nbytes, timeout_s=10.0):
    deadline = time.time() + timeout_s
    buf = bytearray()
    while len(buf) < nbytes and time.time() < deadline:
        chunk = ser.read(nbytes - len(buf))
        if chunk:
            buf.extend(chunk)
    if len(buf) != nbytes:
        return None
    return bytes(buf)

def read_frame_resync(ser, payload_words=784, timeout_s=10.0):
    expected_body = 4 + payload_words * 4  # cycles + data bytes
    deadline = time.time() + timeout_s
    window = bytearray()

    # 1) Scan stream byte-by-byte for MAGIC
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        window += b
        if len(window) > 4:
            window = window[-4:]
        if bytes(window) == MAGIC_BYTES:
            break
    else:
        return None, None, None

    # 2) After MAGIC, read exactly: cycles + payload
    body = read_exact(ser, expected_body, timeout_s=max(0.1, deadline - time.time()))
    if body is None:
        return None, None, None

    # cycles is a 32-bit unsigned little-endian integer
    cycles = struct.unpack('<I', body[:4])[0]

    # payload is interpreted as int32 array
    data = np.frombuffer(body[4:], dtype=np.int32)
    if data.size != payload_words:
        return None, None, None

    return MAGIC, cycles, data

# =============================================================================
# Load one random MNIST test image using PyTorch
# -----------------------------------------------------------------------------
# Returns:
#   - img_np: flattened uint8 array of length 784 in range [0..255]
#   - label: ground-truth label (0..9) if MNIST available, otherwise dummy
# =============================================================================
def get_random_mnist_image():
    try:
        from torchvision import datasets, transforms
        # Load MNIST test split
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        idx = np.random.randint(0, len(test_dataset))
        img_tensor, label = test_dataset[idx]

        # Convert [1, 28, 28] float [0..1] -> uint8 [0..255] flattened [784]
        img_np = (img_tensor.numpy() * 255).astype(np.uint8).flatten()

        print(f"[INFO] Loaded PyTorch MNIST image (Index: {idx}, Label: {label})")
        return img_np, label

    except ImportError:
        # If torchvision is unavailable, fall back to random noise input
        print("[WARN] PyTorch not found. Using Random Noise.")
        img = np.random.randint(0, 255, size=784, dtype=np.uint8)
        return img, 99

    except Exception as e:
        # Any runtime error (e.g., dataset missing) also falls back to noise input
        print(f"[ERROR] {e}")
        img = np.random.randint(0, 255, size=784, dtype=np.uint8)
        return img, 99

# =============================================================================
# ASCII visualization of a 28x28 image
# =============================================================================
def print_ascii_img(img_arr):
    print("\n--- Input Image ---")
    img_2d = img_arr.reshape(28, 28)
    for y in range(28):
        line = ""
        for x in range(28):
            val = img_2d[y, x]
            if val > 200:   line += "@"
            elif val > 100: line += "*"
            elif val > 20:  line += "."
            else:           line += " "
        print(line)
    print("-------------------\n")

# =============================================================================
# Integer hardware-like simulation (Conv1 -> Pool -> Conv2 -> Pool)
# -----------------------------------------------------------------------------
# Key assumptions:
#   - SCALE=256 fixed-point quantization (weights clipped to signed 8-bit range)
#   - Bias scaled by SCALE*SCALE to match accumulator domain
#   - Conv1 output uses ReLU + right shift + clipping to [0..255] before pooling
#   - Conv2 uses ReLU; no extra clip in this debug model (32-bit safe)
# =============================================================================
def simulate_integer_hardware(img_arr, model):
    SCALE = 256

    # -------------------------------------------------------------------------
    # Extract and quantize Conv1 weights and bias
    # Conv1 weights are treated as signed int8 range [-127..127] (stored as int32)
    # Bias is in accumulator domain (SCALE*SCALE)
    # -------------------------------------------------------------------------
    raw_w1 = (model.conv1.weight.detach().numpy() * SCALE).round()
    w1 = np.clip(raw_w1, -127, 127).astype(np.int32)

    b1 = (model.conv1.bias.detach().numpy() * SCALE * SCALE).round().astype(np.int32)

    # -------------------------------------------------------------------------
    # Extract and quantize Conv2 weights and bias
    # -------------------------------------------------------------------------
    raw_w2 = (model.conv2.weight.detach().numpy() * SCALE).round()
    w2 = np.clip(raw_w2, -127, 127).astype(np.int32)

    b2 = (model.conv2.bias.detach().numpy() * SCALE * SCALE).round().astype(np.int32)

    # Input image as int32 (0..255)
    img = img_arr.reshape(28, 28).astype(np.int32)

    # ---------------------------------------------------------
    # Layer 1: Conv1 + ReLU + Shift(10) + Clip(255) + MaxPool
    # ---------------------------------------------------------
    # Padding=1 to keep output at 28x28 before pooling (3x3 kernel)
    img_padded = np.pad(img, ((1,1), (1,1)), 'constant', constant_values=0)

    # After pooling: 28x28 -> 14x14, output channels = 8
    c1_out = np.zeros((8, 14, 14), dtype=np.int32)

    for f in range(8):
        # 1) Convolution over 28x28 output grid
        conv_res = np.zeros((28, 28), dtype=np.int32)
        for y in range(28):
            for x in range(28):
                window = img_padded[y:y+3, x:x+3]
                # Note: w1[f, 0] corresponds to the single input channel
                conv_res[y, x] = np.sum(window * w1[f, 0]) + b1[f]

        # 2) ReLU
        relu_res = np.maximum(conv_res, 0)

        # 3) Right shift to approximate fixed-point scaling normalization
        # 4) Clip to 8-bit unsigned (0..255) for the pool input domain
        shifted_res = relu_res >> 10
        clipped_res = np.clip(shifted_res, 0, 255)

        # 5) MaxPool 2x2 (stride 2): 28x28 -> 14x14
        for y in range(14):
            for x in range(14):
                patch = clipped_res[y*2:y*2+2, x*2:x*2+2]
                c1_out[f, y, x] = np.max(patch)

    # ---------------------------------------------------------
    # Layer 2: Conv2 + ReLU + MaxPool
    # ---------------------------------------------------------
    # Pad conv2 input spatially by 1, keep channel dimension unchanged.
    # c1_out shape: (8, 14, 14) -> padded: (8, 16, 16)
    c2_in = np.pad(c1_out, ((0,0), (1,1), (1,1)), 'constant', constant_values=0)

    # conv2 output before pooling: (16, 14, 14)
    c2_out = np.zeros((16, 14, 14), dtype=np.int32)

    # 1) Conv2 (3x3 over 8 channels)
    for f in range(16):
        for y in range(14):
            for x in range(14):
                acc = 0
                for ch in range(8):
                    window = c2_in[ch, y:y+3, x:x+3]
                    # Multiply unsigned-ish input (0..255) by signed weight (-127..127)
                    acc += np.sum(window * w2[f, ch])
                acc += b2[f]

                # 2) ReLU (no clip here; stored in int32 for debug)
                c2_out[f, y, x] = max(acc, 0)

    # 3) MaxPool 2x2: 14x14 -> 7x7
    final_out = np.zeros((16, 7, 7), dtype=np.int32)
    for f in range(16):
        for y in range(7):
            for x in range(7):
                patch = c2_out[f, y*2:y*2+2, x*2:x*2+2]
                final_out[f, y, x] = np.max(patch)

    return final_out

def idx_to_coord_final(idx):
    # Total 784 = 7*7 pixels * 16 channels
    ch = idx % 16
    pixel_idx = idx // 16
    col = pixel_idx % 7
    row = pixel_idx // 7
    return row, col, ch

def print_all_comparison(fpga, py):
    print("\n" + "="*80)
    print(f"{'Idx':<5} | {'(Row,Col,Ch)':<14} | {'FPGA Val':<12} | {'Python Val':<12} | {'Diff'}")
    print("-" * 80)

    mismatch_count = 0

    for i in range(len(fpga)):
        r, c, ch = idx_to_coord_final(i)

        f_val = fpga[i]
        p_val = py[i]
        diff = int(f_val) - int(p_val)

        # Mark mismatches for quick scanning
        if diff == 0:
            diff_str = "OK"
        else:
            diff_str = f"ERR ({diff})"
            mismatch_count += 1

        print(f"{i:<5} | ({r},{c},{ch:<2}){'':<4} | {f_val:<12} | {p_val:<12} | {diff_str}")

    print("-" * 80)
    return mismatch_count

# =============================================================================
# Main
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1) Load trained model parameters
    # This ensures the Python reference uses the same weights as the FPGA.
    # -------------------------------------------------------------------------
    try:
        model = SmallCNN()
        model.load_state_dict(torch.load("mnist_smallcnn_state_dict.pt", map_location='cpu'))
        model.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # -------------------------------------------------------------------------
    # 2) Open UART connection
    # -------------------------------------------------------------------------
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
        print(f"Successfully connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    print("Press [Enter] to send a new image. Press 'q' to quit.")

    # -------------------------------------------------------------------------
    # 3) Interactive debug loop:
    #   - Generate a random MNIST sample
    #   - Compute Python integer reference (conv1/pool/conv2/pool)
    #   - Send image to FPGA over UART
    #   - Receive FPGA output (7x7x16 = 784 int32 words)
    #   - Print full comparison and mismatch count
    # -------------------------------------------------------------------------
    while True:
        user_input = input(">> Ready? (Enter/q): ")
        if user_input.lower() == 'q':
            break

        # A) Create/load one input image
        img_arr, label_real = get_random_mnist_image()
        print_ascii_img(img_arr)

        # B) Compute reference output (integer hardware-like pipeline)
        ref_out_chw = simulate_integer_hardware(img_arr, model)

        # Flatten reference in (Row, Col, Channel) order to match FPGA streaming order
        ref_flat = ref_out_chw.transpose(1, 2, 0).flatten()

        # C) Send image to FPGA
        # Reset input buffer to reduce chance of reading stale bytes
        ser.reset_input_buffer()
        print(f"[UART] Sending image...")
        ser.write(img_arr.tobytes())

        # D) Receive output frame using resync
        magic, cycles, fpga_data = read_frame_resync(ser, payload_words=784, timeout_s=10.0)
        if fpga_data is None:
            print("[Error] Failed to receive a valid frame (timeout/resync fail).")
            ser.reset_input_buffer()
            continue

        print(f" [FPGA] Cycles: {cycles}, Magic: {hex(magic)}")

        # F) Print full comparison and report mismatches
        errors = print_all_comparison(fpga_data, ref_flat)

        if errors == 0:
            print("\n\033[92m" + "[SUCCESS] Perfect Match! (7x7x16 Output)" + "\033[0m")
        else:
            print(f"\n\033[91m" + f"[FAIL] Total Mismatches: {errors}" + "\033[0m")

    # -------------------------------------------------------------------------
    # Close UART cleanly
    # -------------------------------------------------------------------------
    ser.close()

if __name__ == "__main__":
    main()
