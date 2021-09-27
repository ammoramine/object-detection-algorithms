import fiftyone.zoo as foz
import fiftyone as fo
import os,argparse
dir_file = os.path.dirname(__file__)

#TODO redownload the data
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--overwrite",action='store_true')

    args = parser.parse_args()

    name_dataset = "open-images-v6"
    label_types = ["detections"]
    classes = ["Truck", "Bus"]

    for type_data in ["train","test","validation"]:
        dst_dir = os.path.join(dir_file, f"../../data")
        print(f"importing {type_data} to {dst_dir}")
        kwargs = dict(name=name_dataset,label_types=label_types,classes=classes,dataset_dir= dst_dir)
        kwargs.update(args.__dict__) #update with the arguments specified from argparsers
        foz.download_zoo_dataset(**kwargs)