import pickle
from xgboost import XGBClassifier

# Load the fat pickle
with open('medisense_preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in pickle:", list(data.keys()))

# Save XGBoost model separately in its native format (much smaller)
xgb_model = data['xgb_model']
xgb_model.save_model('xgb_model.json')
print(f"XGBoost model saved separately ✓")

# Remove model objects from pickle — keep only data/metadata
keys_to_remove = ['xgb_model', 'rf_model', 'best_model']
for key in keys_to_remove:
    if key in data:
        del data[key]
        print(f"Removed '{key}' from pickle")

# Save slim pickle
with open('medisense_slim.pkl', 'wb') as f:
    pickle.dump(data, f)

print("\nDone! Check new file sizes:")
import os
print(f"  xgb_model.json      : {os.path.getsize('xgb_model.json')/1e6:.1f} MB")
print(f"  medisense_slim.pkl  : {os.path.getsize('medisense_slim.pkl')/1e6:.1f} MB")
print(f"  rag_knowledge.pkl   : {os.path.getsize('rag_knowledge.pkl')/1e6:.1f} MB")
print(f"  nlp_artifacts.pkl   : {os.path.getsize('nlp_artifacts.pkl')/1e6:.1f} MB")