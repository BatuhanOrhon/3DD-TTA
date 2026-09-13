# Colab ShapeNet Audit Instructions

This script will safely inspect the structure of the downloaded ShapeNetCore v2 dataset, check the available Point-MAE checkpoint, and gather necessary information before we design the HDF5-2048 conversion pipeline.

**Do NOT run the corruption generation script yet.**

Run the following cell in Colab:

```bash
!echo "=== Checking ShapeNet Raw Directory ==="
!ls -lh /content/drive/MyDrive/thesis/dataset/ShapeNetCore_v2_raw | head -n 20

!echo -e "\n=== Checking Inside a Sample Category ZIP ==="
# Attempt to list files in the airplane category zip (02691156)
!zipinfo -1 /content/drive/MyDrive/thesis/dataset/ShapeNetCore_v2_raw/02691156.zip | head -n 20 2>/dev/null || echo "Category ZIP not found or inaccessible"

!echo -e "\n=== Checking Point-MAE ShapeNet Checkpoint ==="
!ls -lh /content/3DD-TTA/Point-MAE/experiments/Point-MAE_ShapeNet/checkpoint_best.pth 2>/dev/null || echo "Point-MAE ShapeNet checkpoint not found"
!sha256sum /content/3DD-TTA/Point-MAE/experiments/Point-MAE_ShapeNet/checkpoint_best.pth 2>/dev/null

!echo -e "\n=== Checking Corruption Script Expected Paths ==="
!head -n 50 /content/3DD-TTA/datasets_mate/create_corrupted_dataset.py | grep -i "ShapeNet" -C 3
```

After running this, please share the raw output (you can just paste the stdout logs into the chat) so we can:
1. Verify the ZIP internal structure.
2. Confirm the Point-MAE ShapeNet checkpoint availability.
3. Understand the exact format `datasets_mate/create_corrupted_dataset.py` expects (it likely expects the specific HDF5 representation).

We will use this information to design the HDF5-2048 conversion path.

### Note on Representation Differences:
- **LION (PC15k)** uses a 15k point representation for ShapeNet.
- **3DD-TTA/Point-MAE (HDF5-2048)** expects an HDF5 2048-point representation.
We will map and document this discrepancy in our next steps.
