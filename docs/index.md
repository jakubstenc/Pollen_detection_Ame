# YOLOv8 Pollen Detection Model Performance

**Last Updated:** 2026-05-10 20:49:36

**Training Images Used:** 19

This page automatically tracks the performance metrics of the latest YOLOv8 Nano instance segmentation training run.

## Model Training Metrics
![Model Training Metrics](assets/results.png)

## Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

## Ground Truth Training Data (What YOLO was taught)
![Ground Truth Training Data (What YOLO was taught)](assets/train_batch0.jpg)

## Validation Predictions (What YOLO guessed)
![Validation Predictions (What YOLO guessed)](assets/val_batch0_pred.jpg)


## Automated Pollen Counting Results

The Bürker grid has been automatically aligned and pollen counted according to the standard **Hemocytometer L-Rule (North/West)**.

> [!NOTE]
> **Counting Protocol**:
> - **Counted (Green)**: Pollen inside the square OR touching the **Top (North)** or **Left (West)** boundaries.
> - **Ignored (Red)**: Pollen touching the **Bottom (South)** or **Right (East)** boundaries.

**📥 [Download Raw Data CSV (pollen_counts.csv)](assets/pollen_counts.csv)**

### 9-Square Statistical Breakdown

|Filename|Total_Counted_Pollen|Average_Per_Square|StdDev_Variability|TL_Count|ML_Count|BL_Count|TC_Count|MC_Count|BC_Count|TR_Count|MR_Count|BR_Count|Ignored_Pollen|Total_Detections|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|O.bi22_prov_dil1a10_2026_05_08_0015(1).JPG|593|65.89|13.41|85|56|55|88|65|60|76|62|46|788|1381|
|O.bi22_prov_dil1a10_2026_05_08_0015(2).JPG|593|65.89|13.41|85|56|55|88|65|60|76|62|46|788|1381|
|O.bi11_prov_dil1a10_2026_05_08_0014.JPG|729|81.0|7.57|81|85|70|75|71|82|92|92|81|1098|1827|
|name_of_your_raw_image.JPG|279|31.0|5.94|38|37|26|34|27|22|24|33|38|266|545|
|O.bi24_prov_dil1a10_2026_05_08_0016.JPG|605|67.22|9.55|71|70|54|76|76|48|63|75|72|892|1497|
|O.bi24_prov_dil1a10_2026_05_08_0016(1).JPG|605|67.22|9.55|71|70|54|76|76|48|63|75|72|892|1497|
|O.bi10_prov_dil1a10_2026_05_08_04_grid.JPG|230|25.56|4.11|27|32|27|19|31|22|21|25|26|256|486|
|O.bi22_prov_dil1a10_2026_05_08_0015.JPG|593|65.89|13.41|85|56|55|88|65|60|76|62|46|788|1381|
|O.bi18_prov_dil1a10_2026_05_08_0017.JPG|782|86.89|17.91|92|70|71|90|88|54|112|110|95|1181|1963|
|O.bi6_scopa_2026_05_08_02.JPG|817|90.78|12.29|108|91|78|99|99|93|83|66|100|1138|1955|
|O.bi10_prov_dil1a10_2026_05_08_04.JPG|279|31.0|5.94|38|37|26|34|27|22|24|33|38|266|545|
|O.bi18_prov_dil1a10_2026_05_08_0017(1).JPG|782|86.89|17.91|92|70|71|90|88|54|112|110|95|1181|1963|

## 🎥 Real-Time Live Camera Tracking

You can bypass static images and run the system directly on a live camera stream! It runs inference directly on the video frames and dynamically snaps the mathematical grid into place.

To run the live stream from your terminal:
```bash
python src/live_counter.py --camera 0 --model runs/pollen_nano_test-3/weights/best.pt
```
*(Change `--camera 0` to a specific index or RTSP stream URL depending on your microscope setup).* 

**Live Features**:
- **`q`** : Quit the live stream.
- **`g`** : Force an instant re-calculation of the mathematical grid.

### visualized_O.bi22_prov_dil1a10_2026_05_08_0015(2).JPG
![visualized_O.bi22_prov_dil1a10_2026_05_08_0015(2).JPG](assets/visualized_O.bi22_prov_dil1a10_2026_05_08_0015(2).JPG)

### visualized_O.bi11_prov_dil1a10_2026_05_08_0014.JPG
![visualized_O.bi11_prov_dil1a10_2026_05_08_0014.JPG](assets/visualized_O.bi11_prov_dil1a10_2026_05_08_0014.JPG)

### visualized_name_of_your_raw_image.JPG
![visualized_name_of_your_raw_image.JPG](assets/visualized_name_of_your_raw_image.JPG)

### visualized_O.bi6_scopa_2026_05_08_02.JPG
![visualized_O.bi6_scopa_2026_05_08_02.JPG](assets/visualized_O.bi6_scopa_2026_05_08_02.JPG)

### visualized_O.bi10_prov_dil1a10_2026_05_08_04_grid.JPG
![visualized_O.bi10_prov_dil1a10_2026_05_08_04_grid.JPG](assets/visualized_O.bi10_prov_dil1a10_2026_05_08_04_grid.JPG)

### visualized_O.bi10_prov_dil1a10_2026_05_08_04.JPG
![visualized_O.bi10_prov_dil1a10_2026_05_08_04.JPG](assets/visualized_O.bi10_prov_dil1a10_2026_05_08_04.JPG)

### visualized_O.bi22_prov_dil1a10_2026_05_08_0015.JPG
![visualized_O.bi22_prov_dil1a10_2026_05_08_0015.JPG](assets/visualized_O.bi22_prov_dil1a10_2026_05_08_0015.JPG)

### visualized_O.bi18_prov_dil1a10_2026_05_08_0017(1).JPG
![visualized_O.bi18_prov_dil1a10_2026_05_08_0017(1).JPG](assets/visualized_O.bi18_prov_dil1a10_2026_05_08_0017(1).JPG)

### visualized_O.bi24_prov_dil1a10_2026_05_08_0016(1).JPG
![visualized_O.bi24_prov_dil1a10_2026_05_08_0016(1).JPG](assets/visualized_O.bi24_prov_dil1a10_2026_05_08_0016(1).JPG)

### visualized_O.bi22_prov_dil1a10_2026_05_08_0015(1).JPG
![visualized_O.bi22_prov_dil1a10_2026_05_08_0015(1).JPG](assets/visualized_O.bi22_prov_dil1a10_2026_05_08_0015(1).JPG)

### visualized_O.bi18_prov_dil1a10_2026_05_08_0017.JPG
![visualized_O.bi18_prov_dil1a10_2026_05_08_0017.JPG](assets/visualized_O.bi18_prov_dil1a10_2026_05_08_0017.JPG)

### visualized_O.bi24_prov_dil1a10_2026_05_08_0016.JPG
![visualized_O.bi24_prov_dil1a10_2026_05_08_0016.JPG](assets/visualized_O.bi24_prov_dil1a10_2026_05_08_0016.JPG)

