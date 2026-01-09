import numpy as np
import torch
import sys

# -----------------------------------------------------------------------------
# Import the training-time model definition so we can load the trained state_dict.
# This script assumes `mnist_train.py` defines `SmallCNN` with:
#   - conv1 (8 filters, 3x3)
#   - conv2 (16 filters, 3x3, 8 input channels)
#   - fc   (10 outputs)
# -----------------------------------------------------------------------------
try:
    from mnist_train import SmallCNN
except ImportError:
    print("[Error] mnist_train.py not found!")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Fixed-point quantization scale.
# - We treat weights as signed int8 in hardware (clipped to [-127, 127])
# - SCALE_FACTOR=256 corresponds to Q8.8-like scaling for weights.
# - Bias is scaled by SCALE_FACTOR*SCALE_FACTOR because it accumulates products.
# -----------------------------------------------------------------------------
SCALE_FACTOR = 256

def write_array(f, name, arr, ctype):
    arr = np.asarray(arr)
    flat = arr.flatten()
    # Emit shape comment to help validate expected layout on the C/HW side.
    f.write(f"// shape: {list(arr.shape)}\n")
    f.write(f"static const {ctype} {name}[{flat.size}] = {{\n    ")
    # Ensure each element is written as an integer literal.
    f.write(", ".join(str(int(x)) for x in flat.tolist()))
    f.write("\n};\n\n")

def main():
    # -------------------------------------------------------------------------
    # Instantiate the model and load trained parameters.
    # If the file is missing, we proceed with random weights (useful only for
    # pipeline bring-up / debugging, NOT for accuracy).
    # -------------------------------------------------------------------------
    model = SmallCNN()
    try:
        sd = torch.load("mnist_smallcnn_state_dict.pt", map_location="cpu")
        model.load_state_dict(sd)
        print("[INFO] Model loaded.")
    except:
        print("[WARN] Using random weights.")

    # Switch to eval mode to ensure deterministic behavior for certain layers
    # (BatchNorm/Dropout etc.). This specific network likely doesn't use them,
    # but keeping this is good practice.
    model.eval()

    def quantize(tensor, is_conv=False):
        """
        Quantize a weight tensor to int32 (later interpreted as int8 range).

        - Multiply by SCALE_FACTOR and round
        - Clip to [-127, 127] to match assumed HW int8 range
        - Return int32 array (stored as int32 in weights.h for convenience)

        For convolution weights:
          PyTorch conv weight shape is typically (Out, In, Ky, Kx).
          The HW reading order in this project is assumed to be:
            [Out, Row, Col, In]  (i.e., channel-last)
          So we transpose to (Out, Ky, Kx, In).
        """
        arr = tensor.detach().numpy()

        if is_conv:
            # HW order: [Out, Row, Col, In] (Channel-Last)
            # PyTorch conv weight: [Out, In, Row, Col]
            # Transpose to: [Out, Row, Col, In]
            arr = arr.transpose(0, 2, 3, 1)

        # Fixed-point scaling and rounding
        q_val = (arr * SCALE_FACTOR).round()

        # Clip to int8-like range (stored as int32 constants in C)
        return np.clip(q_val, -127, 127).astype(np.int32)

    def quantize_bias(tensor, scale):
        return (tensor.detach().numpy() * scale).round().astype(np.int32)

    print("[INFO] Processing Weights with Interleaving...")

    # =========================================================================
    # 1) Conv1 Processing
    # =========================================================================
    # Conv1 weights: shape expected (8, 3, 3, 1) after transpose
    w1 = quantize(model.conv1.weight, is_conv=True)

    # Conv1 bias scaling:
    # - In this design, bias is scaled to match accumulator domain.
    # - Using SCALE_FACTOR * SCALE_FACTOR to match (input_scale * weight_scale).
    b1 = quantize_bias(model.conv1.bias, SCALE_FACTOR * SCALE_FACTOR)

    # Interleaved blob format for HW:
    # For each output filter f:
    #   [w1[f] flattened (Ky*Kx*In), then b1[f]]
    # This matches your C/RTL expectation: "weight first + 1 bias".
    conv1_combined = []
    for f in range(8):
        conv1_combined.extend(w1[f].flatten())
        conv1_combined.append(b1[f])

    # =========================================================================
    # 2) Conv2 Processing
    # =========================================================================
    # Conv2 weights: shape expected (16, 3, 3, 8) after transpose
    w2 = quantize(model.conv2.weight, is_conv=True)

    # Conv2 bias scaling:
    b2 = quantize_bias(model.conv2.bias, SCALE_FACTOR * SCALE_FACTOR)

    # Interleaved blob format for HW:
    # For each output filter f:
    #   [w2[f] flattened (Ky*Kx*In), then b2[f]]
    conv2_combined = []
    for f in range(16):
        conv2_combined.extend(w2[f].flatten())
        conv2_combined.append(b2[f])

    # -------------------------------------------------------------------------
    # 3) FC Processing (RTL Reading Order: Row/Col Major -> Channel Minor)
    # -------------------------------------------------------------------------
    print("[INFO] Packing FC weights (RTL Order: Out, Row, Col, Ch)...")

    # (1) PyTorch fc weight is normally shape (10, 784) after flattening.
    # Here, we reshape it as (10, 16, 7, 7) because:
    #   - The input to FC comes from Conv2 output after pooling: 16 x 7 x 7
    #   - Each class (0..9) has a 16x7x7 kernel-like weight map
    #
    # Original meaning after reshape:
    #   (Out, Channel, Row, Col)
    fc_w_origin = model.fc.weight.data.numpy().reshape(10, 16, 7, 7)

    # (2) Transpose to match RTL reading order:
    #   (Out, Row, Col, Channel)
    # so that when flattened, Channel is the fastest-varying index.
    fc_w_transposed = fc_w_origin.transpose(0, 2, 3, 1)  # Shape: (10, 7, 7, 16)

    # (3) Force contiguous memory layout.
    # This prevents unexpected flatten order due to non-contiguous strides.
    # This is critical for bit-accurate HW/SW matching.
    fc_w_contiguous = np.ascontiguousarray(fc_w_transposed)

    # (4) Quantize FC weights to int8-like domain stored as int32 constants.
    fc_w_q = (fc_w_contiguous * SCALE_FACTOR).round()
    fc_w_q = np.clip(fc_w_q, -127, 127).astype(np.int32)

    # (5) Flatten & Pack
    # For each output class:
    #   - Flatten 7*7*16 = 784 weights in (Row, Col, Ch) order
    #   - Pack 4 signed 8-bit weights into one 32-bit word (little-endian)
    #
    # Result:
    #   - 784 weights / 4 = 196 words per class
    #   - 10 classes -> 1960 words total
    fc_packed_list = []

    for out_idx in range(10):  # Class 0~9
        flat_w = fc_w_q[out_idx].flatten()  # 7*7*16 = 784 weights

        # Sanity interpretation:
        # flat_w[0..3] corresponds to (Row0, Col0) with (Ch0..Ch3),
        # matching the RTL consumption order.
        packed_row = []

        # 4 bytes -> 1 word (Little Endian)
        # Each flat_w[i] is in int32 but conceptually holds an int8 range [-127..127].
        # We mask with 0xFF to keep the low 8-bit two's complement representation.
        for i in range(0, 784, 4):
            val = (flat_w[i] & 0xFF) | \
                  ((flat_w[i+1] & 0xFF) << 8) | \
                  ((flat_w[i+2] & 0xFF) << 16) | \
                  ((flat_w[i+3] & 0xFF) << 24)
            packed_row.append(val)

        # Append packed words for this output class
        fc_packed_list.extend(packed_row)  # 196 words

    # FC Bias
    # Bias is quantized in accumulator domain (SCALE_FACTOR*SCALE_FACTOR).
    bf = quantize_bias(model.fc.bias, SCALE_FACTOR * SCALE_FACTOR)

    # =========================================================================
    # 4) Emit C header file (weights.h)
    # =========================================================================
    # The output header is consumed by MicroBlaze host code.
    # Arrays:
    #   - conv1_params: interleaved weights + bias per filter
    #   - conv2_params: interleaved weights + bias per filter
    #   - fc_w_packed : packed int8 weights into 32-bit words, RTL order
    #   - fc_b        : FC bias (int32)
    with open("weights.h", "w", encoding="utf-8") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        write_array(f, "conv1_params", conv1_combined, "int32_t")
        write_array(f, "conv2_params", conv2_combined, "int32_t")

        # Use packed FC weights for RTL-friendly reading.
        # The symbol name must match what the C firmware expects.
        write_array(f, "fc_w_packed", fc_packed_list, "int32_t")

        write_array(f, "fc_b", bf, "int32_t")

        f.write("#endif\n")

    print("[SUCCESS] 'weights.h' generated with PACKED FC weights!")

if __name__ == "__main__":
    main()
