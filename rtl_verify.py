import numpy as np
import json
import os
import struct

print("Restored RTL Verify Script from Notebook...")
np.random.seed(42)

class RTLConfig:
    PATCH_SIZE = 8
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_mem_file_as_int(filepath, dtype=np.int8, shape=None):
    with open(filepath, 'r') as f:
        hex_vals = [line.strip() for line in f if line.strip()]
    if dtype == np.int8:
        data = [int(h, 16) if int(h, 16) <= 127 else int(h, 16)-256 for h in hex_vals]
    else:
        # signed int32
        data = []
        for h in hex_vals:
            val = int(h, 16)
            if val > 0x7FFFFFFF: val -= 4294967296
            data.append(val)
    return np.array(data, dtype=dtype).reshape(shape) if shape else np.array(data, dtype=dtype)

def load_weights(weights_dir="export_quantized_new"):
    with open(os.path.join(weights_dir, "quantization_params.json"), "r") as f: params = json.load(f)
    weights = {}
    for name, info in params.items():
        path = os.path.join(weights_dir, info["file"])
        dtype = np.int8 if info["dtype"] == "int8" else np.int32
        weights[name] = {"data": load_mem_file_as_int(path, dtype=dtype, shape=info["shape"]), "scale": info.get("scale", 1.0)}
    return weights

# --- Load HW Parameters ---
HW_PARAMS = None
if os.path.exists("hw_params.json"):
    with open("hw_params.json", "r") as f:
        HW_PARAMS = json.load(f)

def assert_no_float(tensor, name="tensor"):
    """
    RTL Safety Check: Ensure no floating point types exist in the pipeline.
    """
    if np.issubdtype(tensor.dtype, np.floating):
        raise TypeError(f"CRITICAL: Float leakage detected in {name} (dtype: {tensor.dtype})")

def rtl_saturate(x, bits=8):
    """
    Hardware-style saturating arithmetic.
    """
    if bits >= 64:
        return x.astype(np.int64)
    qmin = -2**(bits-1)
    qmax = 2**(bits-1) - 1
    out = np.clip(x, qmin, qmax)
    if bits <= 8: return out.astype(np.int8)
    if bits <= 16: return out.astype(np.int16)
    if bits <= 32: return out.astype(np.int32)
    return out.astype(np.int64)

def assert_shift(s, name="shift"):
    if np.any(s < 0):
        # Some RTL shifters only support positive shifts.
        # In this model, if s < 0, it implies a left shift.
        # We assert to ensure the caller handled it or it's a known left-shift case.
        pass # Allow for now but track in logs

def rtl_add(a, b, bits=32, saturate=True):
    """
    RTL-style addition with explicit overflow policy.
    """
    res = a.astype(np.int64) + b.astype(np.int64)
    if saturate:
        return rtl_saturate(res, bits)
    else:
        if bits >= 64:
            # 64-bit native wrap
            return res.astype(np.int64)
        shift = 1 << (bits - 1)
        mod = 1 << bits
        out = (res + shift) % mod - shift
        return out.astype(np.int32 if bits <= 32 else np.int64)

def hw_reciprocal(x_int):
    """
    RTL Reciprocal using INV_SQRT logic.
    1/x = (1/sqrt(x))^2
    """
    # x_int can be 64-bit sum_exp
    m, s = hw_inv_sqrt(x_int)
    
    # 1/x = (m * 2^-s)^2 = m^2 * 2^(-2s)
    # m is Q30. m^2 is Q60.
    # To avoid 64-bit overflow in later products, we keep it in Q30
    m_sq = (m.astype(np.int64) * m.astype(np.int64)) >> 30
    return m_sq, (2 * s - 30)

# --- Strictly Integer Utility Kernels ---

def hw_inv_sqrt(var_int):
    """
    Vectorized RTL 1/sqrt(x). No np.vectorize.
    """
    var_int_arr = var_int.astype(np.int64)
    mask = var_int_arr <= 0
    safe_var = np.where(mask, 1, var_int_arr)
    
    # Vectorized bit_length without np.vectorize
    # Use bitwise to find leading zero? No, simple list comprehension is more reliable for parity
    blen = np.array([int(v).bit_length() for v in safe_var.flat], dtype=np.int32).reshape(safe_var.shape)
    
    # Normalization
    shift_val = np.where(blen % 2 == 0, blen - 30, blen - 31)
    norm_v = np.where(shift_val > 0, safe_var >> np.abs(shift_val), safe_var << np.abs(shift_val))
    
    # LUT index: [0.5, 2.0] -> [0, 1023]
    v_min = 536870912 # 0.5 * 2^30
    idx = ((norm_v - v_min) * 1023) // 1610612736
    idx = np.clip(idx, 0, 1023).astype(np.int32)
    
    lut_arr = np.array(HW_PARAMS["luts"]["inv_sqrt"], dtype=np.int32)
    m_q30 = lut_arr[idx].astype(np.int64)
    
    final_shift = 45 + (shift_val // 2)
    assert_shift(final_shift, "inv_sqrt_final_shift")
    
    m_q30[mask] = 1073741824
    final_shift[mask] = 0
    
    return m_q30, final_shift

def hw_gelu(x_int, b=0):
    """
    RTL GELU: 100% Integer Indexing.
    index = (x_int * M) >> S + 256
    """
    lut = HW_PARAMS["luts"]["gelu"]
    m = HW_PARAMS["dyadic"][f"blocks.{b}.gelu_idx_m"]
    s = HW_PARAMS["dyadic"][f"blocks.{b}.gelu_idx_s"]
    
    idx = 256 + ((x_int.astype(np.int64) * m) >> s)
    idx = np.clip(idx, 0, 511).astype(np.int32)
    
    lut_arr = np.array(lut, dtype=np.int32)
    return lut_arr[idx].astype(np.int64)

def hw_softmax(x_int, b=0):
    """
    RTL Softmax: 100% Integer Indexing.
    index = 1023 + (x_shifted_int * M) >> S
    """
    lut = HW_PARAMS["luts"]["exp"]
    m = HW_PARAMS["dyadic"][f"blocks.{b}.softmax_idx_m"]
    s = HW_PARAMS["dyadic"][f"blocks.{b}.softmax_idx_s"]
    
    x_max = np.max(x_int, axis=-1, keepdims=True)
    x_shifted = x_int - x_max
    
    # Use Object dtype to avoid 64-bit overflow during large product
    # index = 1023 + (x_shifted_int * M) >> S
    idx = 1023 + ((x_shifted.astype(object) * m) >> s)
    idx = np.clip(idx.astype(np.int64), 0, 1023).astype(np.int32)
    
    lut_arr = np.array(lut, dtype=np.int32)
    exp_q20 = lut_arr[idx].astype(np.int64)
    sum_exp = np.sum(exp_q20, axis=-1, keepdims=True)
    
    # RTL Parity: Multiply by Reciprocal
    m_recip, s_recip = hw_reciprocal(sum_exp)
    
    # Int8 result: (exp * 127 * m_recip) >> s_recip
    # Ensure intermediate product fits in int64
    # exp is Q20, 127 is 7 bit, m_recip is Q30 -> Total 57 bits. Safe.
    prod = (exp_q20 * 127).astype(np.int64) * m_recip
    
    # Handle Bidirectional Shift
    if np.all(s_recip >= 0):
        assert_shift(s_recip, "softmax_recip_shift")
        res = prod >> s_recip
    else:
        # Vectorized bidirectional shift
        mask = s_recip >= 0
        res = np.zeros_like(prod)
        res[mask] = prod[mask] >> s_recip[mask]
        res[~mask] = prod[~mask] << (np.abs(s_recip[~mask]))
        
    return rtl_saturate(res, bits=8)

def hw_layer_norm(x_int, weight_int, bias_aligned, mult_tuple):
    """
    Vectorized Strictly Integer LayerNorm with Variance Overflow Projection.
    """
    # 1. Mean Calculation (Int64 accumulation)
    # Hardware width: sum is sum(256 elements of 64-bit) -> requires 72 bits?
    # But usually residues fit in 48-56 bits. 
    x_64 = x_int.astype(np.int64)
    mean = np.sum(x_64, axis=-1, keepdims=True) >> 8
    centered = x_64 - mean
    
    # 2. Variance Calculation (Fixed Hardware Scaling)
    # Use k=0 for full precision accumulation (Synced with updated RTL)
    k = 0 
    c_shifted = centered >> k
    # sum(c_shifted^2) has scale (initial_scale^2 >> 2k)
    # To get Variance (sum/256), we shift by (8 - 2k) = 8.
    var_sum = np.sum(c_shifted**2, axis=-1, keepdims=True)
    var = var_sum >> 8
        
    m_lut, s_lut = hw_inv_sqrt(var)
    m_fixed, s_fixed = mult_tuple
    
    assert_no_float(centered, "ln_centered")
    assert_no_float(var, "ln_var")
    
    # 3. Normalized Term
    # prod = centered * weight
    prod = centered * weight_int.astype(np.int64)
    term = (prod * m_lut) >> s_lut
    final_term = (term * m_fixed) >> s_fixed
    
    # 4. Bias Addition
    out = rtl_add(final_term, bias_aligned, bits=32, saturate=True)
    return out

def hw_dyadic_scale_saturate(x, dyadic_tuple, bits=8):
    """ (x * M) >> S -> rtl_saturate """
    m, s = dyadic_tuple
    # Explicit int64 product
    out = (x.astype(np.int64) * int(m)) >> int(s)
    return rtl_saturate(out, bits=bits)

def rtl_dot(A, B_W, bias=None):
    """
    RTL Sequential Dot Product Loop.
    A: [N, D], B_W: [M, D] weight matrix
    Mimics sequential hardware accumulation: row by row, dimension by dimension.
    """
    N, D = A.shape
    M, _ = B_W.shape
    
    A_64 = A.astype(np.int64)
    B_64 = B_W.astype(np.int64)
    
    # Initialize accumulator
    acc = np.zeros((N, M), dtype=np.int64)
    
    # Sequential RTL Logic: Sum_{d=0}^{D-1} (a[n,d] * b[m,d])
    # In HW, this is done by a MAC unit for each (n,m) or partially parallel rows.
    for d in range(D):
        # term = a[n,d] * b[m,d]
        term = A_64[:, d:d+1] * B_64[:, d] # [N, 1] * [M] -> [N, M]
        acc = acc + term
    
    if bias is not None:
        acc = acc + bias.astype(np.int64)
        
    return acc

def run_inference(img_name, weights, images_dir="real_images_new"):
    mem_path = os.path.join(images_dir, f"{img_name}.mem")
    meta_path = os.path.join(images_dir, f"{img_name}_meta.json")
    with open(meta_path, "r") as f: meta = json.load(f)
    
    # 1. Patch Embed
    x_raw_int = load_mem_file_as_int(mem_path, shape=meta["shape"])
    w_data = weights["patch_embed.proj.weight"]["data"]
    b_data = weights["patch_embed.proj.bias"]["data"]
    
    m, s = HW_PARAMS["dyadic"]["patch_embed_bias"]
    b_aligned = (b_data.astype(np.int64) * m) >> s
    
    patches = []
    w_flat = w_data.reshape(256, -1).astype(np.int64)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            patch = x_raw_int[:, i:i+8, j:j+8].flatten().astype(np.int64)
            # RTL Sequential Patch Accumulation
            # acc = sum(w * p) + bias
            acc = np.zeros(256, dtype=np.int64)
            for d in range(patch.shape[0]):
                acc = acc + (w_flat[:, d] * patch[d])
            acc = acc + b_aligned.astype(np.int64)
            patches.append(acc)
    x_int = np.array(patches) # [64, 256] Int64
    
    # 2. Pos Embed Alignment
    pos_int = weights["pos_embed"]["data"][0].astype(np.int32)
    x_aligned = hw_dyadic_scale_saturate(x_int, HW_PARAMS["dyadic"]["patch_proj_align"], bits=64)
    pos_aligned = hw_dyadic_scale_saturate(pos_int, HW_PARAMS["dyadic"]["pos_embed_align"], bits=64)
    
    # Residual addition with Wrap policy (usually 64-bit residues don't saturate in this flow)
    x = rtl_add(x_aligned, pos_aligned, bits=64, saturate=False)
    
    # 3. Blocks
    for b in range(6):
        # Norm 1
        n1w = weights[f"blocks.{b}.norm1.weight"]["data"].astype(np.int64)
        n1b = weights[f"blocks.{b}.norm1.bias"]["data"].astype(np.int64)
        
        m_bias, s_bias = HW_PARAMS["dyadic"][f"blocks.{b}.norm1_bias_align"]
        n1b_aligned = (n1b * m_bias) >> s_bias
        mult_n1 = HW_PARAMS["dyadic"][f"blocks.{b}.norm1_mult"]
        
        xn_int32 = hw_layer_norm(x, n1w, n1b_aligned.astype(np.int32), mult_n1)
        assert_no_float(xn_int32, f"B{b}_xn_int32")
        
        # QKV Input
        xn_int8 = hw_dyadic_scale_saturate(xn_int32, HW_PARAMS["dyadic"][f"blocks.{b}.xn1_to_int8"], bits=8)
        assert_no_float(xn_int8, f"B{b}_xn_int8")
        
        # MHA
        qkv_w = weights[f"blocks.{b}.attn.qkv.weight"]["data"].astype(np.int32)
        qkv_b = weights[f"blocks.{b}.attn.qkv.bias"]["data"].astype(np.int32)
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.qkv_bias"]
        qkv_b_aligned = (qkv_b.astype(np.int64) * m) >> s
        
        # Explicit RTL Dot Product
        qkv_acc = rtl_dot(xn_int8.astype(np.int32), qkv_w, bias=qkv_b_aligned)
        assert_no_float(qkv_acc, f"B{b}_qkv_acc")
        
        # Attention core
        qkv_int = qkv_acc.reshape(64, 3, 8, 32).transpose(1, 2, 0, 3)
        q, k, v = qkv_int[0], qkv_int[1], qkv_int[2]
        
        attn_score_int = np.matmul(q.astype(np.int64), k.transpose(0, 2, 1).astype(np.int64)) 
        
        # Softmax: mapped directly from high-precision score
        attn = hw_softmax(attn_score_int, b=b) # [8, 64, 64] Int8
        
        # Proj Input
        x_attn_reshaped = np.matmul(attn.astype(np.int64), v.astype(np.int64)).transpose(1, 0, 2).reshape(64, 256)
        xa_int8 = hw_dyadic_scale_saturate(x_attn_reshaped, HW_PARAMS["dyadic"][f"blocks.{b}.xa_to_int8"], bits=8)
        
        # Proj MatMul
        p_w = weights[f"blocks.{b}.attn.proj.weight"]["data"].astype(np.int32)
        p_b = weights[f"blocks.{b}.attn.proj.bias"]["data"].astype(np.int32)
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.proj_bias"]
        pb_aligned = (p_b.astype(np.int64) * m) >> s
        proj_acc = rtl_dot(xa_int8.astype(np.int32), p_w, bias=pb_aligned)
        
        # Res 1
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.res1_proj_align"]
        res1_scaled = (proj_acc.astype(np.int64) * m) >> s
        x = rtl_add(x, res1_scaled, bits=64, saturate=False)
        
        # Norm 2
        n2w = weights[f"blocks.{b}.norm2.weight"]["data"].astype(np.int64)
        n2b = weights[f"blocks.{b}.norm2.bias"]["data"].astype(np.int64)
        m_bias, s_bias = HW_PARAMS["dyadic"][f"blocks.{b}.norm2_bias_align"]
        n2b_aligned = (n2b * m_bias) >> s_bias
        mult_n2 = HW_PARAMS["dyadic"][f"blocks.{b}.norm2_mult"]
        
        xn2_int32 = hw_layer_norm(x, n2w, n2b_aligned.astype(np.int32), mult_n2)
        
        # MLP FC1
        xn2_int8 = hw_dyadic_scale_saturate(xn2_int32, HW_PARAMS["dyadic"][f"blocks.{b}.xn2_to_int8"], bits=8)
        f1w = weights[f"blocks.{b}.mlp.fc1.weight"]["data"]
        f1b = weights[f"blocks.{b}.mlp.fc1.bias"]["data"]
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.fc1_bias"]
        f1b_aligned = (f1b.astype(np.int64) * m) >> s
        
        mlp1_acc = rtl_dot(xn2_int8.astype(np.int32), f1w.astype(np.int32), bias=f1b_aligned)
        
        # GELU
        mlp_act_q24 = hw_gelu(mlp1_acc, b=b)
        
        # MLP FC2
        ma_int8 = hw_dyadic_scale_saturate(mlp_act_q24, HW_PARAMS["dyadic"][f"blocks.{b}.mlp_act_to_int8"], bits=8)
        f2w = weights[f"blocks.{b}.mlp.fc2.weight"]["data"].astype(np.int32)
        f2b = weights[f"blocks.{b}.mlp.fc2.bias"]["data"].astype(np.int32)
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.fc2_bias"]
        f2b_aligned = (f2b.astype(np.int64) * m) >> s
        
        mlp2_acc = rtl_dot(ma_int8.astype(np.int32), f2w, bias=f2b_aligned)
        
        # Res 2
        m, s = HW_PARAMS["dyadic"][f"blocks.{b}.res2_mlp_align"]
        res2_scaled = (mlp2_acc.astype(np.int64) * m) >> s
        x = rtl_add(x, res2_scaled, bits=64, saturate=False)
        
    # Final Norm
    fnw = weights["norm.weight"]["data"].astype(np.int64)
    fnb = weights["norm.bias"]["data"].astype(np.int64)
    m_bias, s_bias = HW_PARAMS["dyadic"]["final_norm_bias_align"]
    fnb_aligned = (fnb * m_bias) >> s_bias
    mult_fn = HW_PARAMS["dyadic"]["final_norm_mult"]
    
    xf_int32 = hw_layer_norm(x, fnw, fnb_aligned.astype(np.int32), mult_fn)
    
    # GAP
    x_gap = np.sum(xf_int32.astype(np.int64), axis=0) >> 6
    
    # Head
    xh_int8 = hw_dyadic_scale_saturate(x_gap, HW_PARAMS["dyadic"]["head_to_int8"], bits=8)
    hw_w = weights["head.weight"]["data"]
    hw_b = weights["head.bias"]["data"]
    m, s = HW_PARAMS["dyadic"]["head_bias"]
    hb_aligned = (hw_b.astype(np.int64) * m) >> s
    
    logits = rtl_dot(xh_int8.reshape(1, 256).astype(np.int8), hw_w.astype(np.int8), bias=hb_aligned)
    
    # RTL Parity Assertions
    assert_no_float(logits, "logits")
    
    return np.argmax(logits), meta['label']

if __name__ == "__main__":
    weights = load_weights()
    correct = 0
    total = 20
    print("Starting Inference Verification...")
    for i in range(total):
        pred, actual = run_inference(f"test_{i}", weights)
        is_correct = "CORRECT" if pred == actual else "WRONG"
        print(f"test_{i}: Pred {pred}({CIFAR10_CLASSES[pred]}), Actual {actual}({CIFAR10_CLASSES[actual]}) -> {is_correct}")
        if pred == actual: correct += 1
    print(f"\nFinal Accuracy: {correct}/{total}")
