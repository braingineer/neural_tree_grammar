import yaml
from tqdm import tqdm


def make():
    config = Blob("../pipeline_settings.conf")
    data_files = ["{}_{}.pkl".format(config.data_prefix, split) 
                    for split in ("train", "dev", "test")]

    for data_file in tqdm(data_files, desc='data file', position=0):
        with open(join(config.processed_data_dir, data_file)) as fp:
            data = pickle.load(fp)
        dataset = []
        data_name = data_file.replace(".pkl","")
        for datum in tqdm(data, desc="{} data".format(data_name), position=1):
            full_tree = rollout(datum)
            tree_features = extract_features(full_tree)
            ## note: this needs to be deprecated by fixing this in the induction
            #roll_features = [(tempfix(a), tempfix(b), c, d, e) 
            #                    for a,b,c,d,e in full_tree.modo_roll_features()]
            dataset.extend(tree_features)
        with open(join(data_fp, data_file.replace(".pkl", "_fergus_N_supertags.pkl")), "w") as fp:
            pickle.dump(dataset, fp)