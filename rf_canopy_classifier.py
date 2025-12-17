import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import subprocess
import glob

# Get labeled NAIP tiles from directory
naip_files = sorted(
    glob.glob("data/patches/labeled/**/**/*.tif", recursive=True)
)
naip_files = [f for f in naip_files if "_treemask" not in f]


# And get their corresponding masks
mask_files = sorted(
    glob.glob("data/patches/labeled/**/**/*_treemask.tif", recursive=True)
)

print(f"Total NAIP patches found: {len(naip_files)}")
print("Example NAIP patches:")
for f in naip_files[:5]:   # show first 5
    print("  ", f)

print("\nTotal labeled mask patches found: {0}".format(len(mask_files)))
print("Example mask patches:")
for f in mask_files[:5]:   # show first 5
    print("  ", f)



X_all, y_all = [], []

for naip_path, mask_path in zip(naip_files, mask_files):
    with rasterio.open(naip_path) as src:
        img = src.read()  # (bands, H, W)
        img = np.moveaxis(img, 0, -1)  # (H, W, bands)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # single-band mask

    # Flatten - each patch is flattened to pixels
    X = img.reshape(-1, img.shape[2]) #NAIP patch flattened into pixels (x in X includes freatures e.g. [r,g,b,nir])
    y = mask.flatten()  #labeled mask flattened into pixels

    # Remove nodata if any
    valid_idx = y >= 0
    X_all.append(X[valid_idx])
    y_all.append(y[valid_idx])



# Stack flattenned patches together
X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

print("Training features:", X_all.shape)
print("Training labels:", y_all.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))


unlabeled_files = sorted(glob.glob("data/patches/NAIP_512x512/**/**/*.tif", recursive=True))

# Filter out labeled and pyramid-level-1 patches
unlabeled_files = [
    f for f in unlabeled_files 
    if "labeled" not in f and "/1/" not in f
]

for naip_path in unlabeled_files:
    with rasterio.open(naip_path) as src:
        img = src.read()
        profile = src.profile
        img = np.moveaxis(img, 0, -1)

    # Flatten
    X_new = img.reshape(-1, img.shape[2])

    # Predict
    y_pred_new = rf.predict(X_new)
    pred_mask = y_pred_new.reshape(img.shape[0], img.shape[1])

     # ---- PAUSE FOR CHECK ----
    if naip_path == unlabeled_files[0]: # only pause for the first patch
        print(f"First patch prediction shape: {pred_mask.shape}")
        # Save temporary image for visual check
        plt.imshow(pred_mask, cmap='Greens')
        plt.title("First Predicted Mask")
        plt.axis('off')
        plt.savefig("temp_first_prediction.png", dpi=150)
        plt.close()
        # Automatically open the PNG on Mac
        subprocess.run(["open", "temp_first_prediction.png"])
        input("Check temp_first_prediction.png. Press Enter to continue with all patches...")

    # Save prediction
    out_path = naip_path.replace("NAIP_512x512", "rf_predictions")
    profile.update(count=1, dtype="uint8")

    # ---- Ensure the output folder exists ----
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_mask.astype("uint8"), 1)

    print(f"Saved prediction: {out_path}")
