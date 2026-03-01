import torch
import onnx
from src.model import LiquidMalwareDetector

def export_safe_mode():
    print("🔄 Loading Model...")
    model = LiquidMalwareDetector(input_features=3, hidden_units=64)
    model.eval()

    # 1. Create Dummy Input (Fixed Size: Batch=1, Seq=50, Feat=3)
    # This MUST match what your Android App sends exactly.
    dummy_packet = torch.randn(1, 50, 3) 
    dummy_time   = torch.randn(1, 50, 1)

    print("📦 Exporting to ONNX...")
    
    # 2. EXPORT WITH DYNAMIC BATCH AXES
    onnx_path = "liquid_defense.onnx"
    torch.onnx.export(
        model,
        (dummy_packet, dummy_time),
        onnx_path,
        input_names=["packet_features", "time_deltas"],
        output_names=["malware_probability"],
        opset_version=14,  # Modern opset for broader compatibility
        dynamic_axes={
            "packet_features": {0: "batch_size"},
            "time_deltas": {0: "batch_size"},
            "malware_probability": {0: "batch_size"}
        }
    )

    print("🔄 Downgrading IR Version for older Android compatibility...")
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)

    print(f"✅ SUCCESS! Model saved and downgraded as '{onnx_path}'")

if __name__ == "__main__":
    export_safe_mode()