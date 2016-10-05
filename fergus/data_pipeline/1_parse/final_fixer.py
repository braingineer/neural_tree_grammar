import glob

for file in glob.glob("preprocessed/*.mrg"):
    trees = []
    mode = "pass"
    bucket = ""
    print("starting ", file)
    with open(file) as fp: 

        for line in fp.readlines():
            if "Original tree" in line:
                mode = "pass"
            elif "Tree transformed" in line:
                mode = "accum"
            elif "-------------" in line:
                trees.append(bucket)
                bucket = ""
            elif mode == "accum":
                bucket += line
    newf = file.replace("preprocessed", "final")
    with open(newf, "w") as fp:
        for t in trees:
            fp.write(t)
