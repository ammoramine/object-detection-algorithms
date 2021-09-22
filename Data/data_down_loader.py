import fiftyone.zoo as foz
import fiftyone as fo
import os
dir_file = os.path.dirname(__file__)


for type_data in ["train","test","validation"]:
    dst_dir = os.path.join(dir_file,f"{type_data}_data")
    print(f"importing {type_data} to {dst_dir}")
    foz.download_zoo_dataset("open-images-v6",split=f"{type_data}",label_types=["detections"],classes=["Truck","Bus"],dataset_dir = dst_dir)