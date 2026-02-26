import glob

# folder where your Roboflow YOLO labels are
label_files = glob.glob("data/valid/labels/*.txt")

for file in label_files:
    with open(file, "r") as f:
        lines = f.readlines()
    # replace first number in each line with 0
    new_lines = ["0" + line[line.find(" "):] for line in lines]
    with open(file, "w") as f:
        f.writelines(new_lines)