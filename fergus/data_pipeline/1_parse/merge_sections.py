import glob

for folder in glob.glob("*/"):
    if not folder.isdigit(): continue
    combined = ""
    for file in glob.glob(folder+"/*.mrg"):
        with open(file) as fp:
            combined += "\n" + fp.read()
    with open("merged/"+folder.replace("/",".mrg"), "w") as fp:
        fp.write(combined)