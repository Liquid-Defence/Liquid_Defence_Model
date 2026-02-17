import torch
from src.model import LiquidMalwareDetector

def export_safe_mode():
    print("🔄 Loading Model...")
    model = LiquidMalwareDetector(input_features=3, hidden_units=64)
    model.eval()

    # 1. Create Dummy Input (Fixed Size: Batch=1, Seq=50, Feat=3)
    # This MUST match what your Android App sends exactly.
    dummy_packet = torch.randn(1, 50, 3) 
    dummy_time   = torch.randn(1, 50, 1)

    print("📦 Exporting to ONNX (Safe Mode: Opset 11, Fixed Size)...")
    
    # 2. EXPORT WITHOUT DYNAMIC AXES
    torch.onnx.export(
        model,
        (dummy_packet, dummy_time),
        "liquid_defense.onnx",
        input_names=["packet_features", "time_deltas"],
        output_names=["malware_probability"],
        opset_version=11  # <--- Golden Standard for Mobile
        # dynamic_axes argument is REMOVED for safety
    )

    print("✅ SUCCESS! Safe 'liquid_defense.onnx' saved.")

if __name__ == "__main__":
    export_safe_mode()