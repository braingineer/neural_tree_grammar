import subprocess

command = "~/research/code/stanford-parser/bcm_transformer.sh merged/{0} > sfprocced/{0}"

import glob

for file in glob.glob("merged/*.mrg"):
    new_c = command.format(file.replace("merged/",""))
    print("running.. {}".format(new_c))
    subprocess.call(new_c, shell=True)

