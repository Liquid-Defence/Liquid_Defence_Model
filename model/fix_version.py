import onnx

# 1. Load the model you just exported
# (Make sure "liquid.onnx" matches the name of your file!)
model_path = "liquid_defense.onnx" 
onnx_model = onnx.load(model_path)

print(f"Original IR Version: {onnx_model.ir_version}")

# 2. Force the downgrade to Version 9
onnx_model.ir_version = 9

# 3. Save it with the name your Android app expects
final_name = "liquid_defense.onnx"
onnx.save(onnx_model, final_name)

print(f"✅ SUCCESS! Model downgraded and saved as '{final_name}'")