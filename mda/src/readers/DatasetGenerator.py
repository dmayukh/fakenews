import json
from tqdm import tqdm
from drqa import retriever
import os


class DatasetGenerator:
    """
    self.dataset_root = '/local/fever-common/'
    self.out_dir = 'working/data/out/'
    self.tdidf_npz_file = self.dataset_root + 'data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    """
    def __init__(self, dataset_root, out_dir):
        self.dataset_root = dataset_root
        self.out_dir = out_dir
        self.tdidf_npz_file = self.dataset_root + 'data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
        self.ranker = retriever.get_class('tfidf')(tfidf_path=self.tdidf_npz_file)

    def generate_nei_evidences(self, split, k=5):
        if not os.path.exists(self.out_dir):
            print("Creating directory {}".format(self.out_dir))
            os.makedirs(self.out_dir)
        print("Writing data to {}".format("{0}/{1}.ns.pages.p{2}.jsonl".format(self.out_dir, split, k)))
        with open(self.dataset_root + "data/fever-data/{0}.jsonl".format(split), "r") as f_in:
            with open(self.out_dir + "{0}.ns.pages.p{1}.jsonl".format(split, k), "w+") as f_out:
                for line in tqdm(f_in.readlines()):
                    line = json.loads(line)
                    if line["label"] == "NOT ENOUGH INFO":
                        doc_names, doc_scores = self.ranker.closest_docs(line['claim'], k)
                        pp = list(doc_names)

                        for idx, evidence_group in enumerate(line['evidence']):
                            for evidence in evidence_group:
                                if idx < len(pp):
                                    evidence[2] = pp[idx]
                                    evidence[3] = -1

                    f_out.write(json.dumps(line) + "\n")