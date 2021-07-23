import json
from tqdm import tqdm
from drqa import retriever
import os
import numpy as np
from drqascripts.retriever.build_tfidf_lines import OnlineTfidfDocRanker
import math
from multiprocessing.pool import ThreadPool
from drqa.retriever import DocDB, utils
import random
import os


class SimpleRandom():
    instance = None

    def __init__(self, seed):
        self.seed = seed
        self.random = random.Random(seed)

    def next_rand(self, a, b):
        return self.random.randint(a, b)

    @staticmethod
    def get_instance():
        if SimpleRandom.instance is None:
            SimpleRandom.instance = SimpleRandom(SimpleRandom.get_seed())
        return SimpleRandom.instance

    @staticmethod
    def get_seed():
        return int(os.getenv("RANDOM_SEED", 12459))

class RankArgs:
    def __init__(self):
        self.ngram = 2
        self.hash_size = int(math.pow(2,24))
        self.tokenizer = "simple"
        self.num_workers = None

class FeverDocDB(DocDB):

    def __init__(self,path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

class DatasetGenerator:
    """
    self.dataset_root = '/local/fever-common/'
    self.out_dir = 'working/data/out/'
    self.tdidf_npz_file = self.dataset_root + 'data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    """
    def __init__(self, dataset_root, out_dir, database_path, init_ranker=True):
        self.dataset_root = dataset_root
        self.out_dir = out_dir
        if init_ranker:
            self.tdidf_npz_file = self.dataset_root + '/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
            self.ranker = retriever.get_class('tfidf')(tfidf_path=self.tdidf_npz_file)
        else:
            self.ranker = None
        self.args = RankArgs()
        self.database_path = database_path
        self.database = FeverDocDB(self.database_path)

    def generate_nei_evidences(self, split, k=5):
        if not os.path.exists(self.out_dir):
            print("Creating directory {}".format(self.out_dir))
            os.makedirs(self.out_dir)
        print("Writing data to {}".format("{0}/{1}.ns.pages.p{2}.jsonl".format(self.out_dir, split, k)))
        with open(self.dataset_root + "/fever-data/{0}.jsonl".format(split), "r") as f_in:
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

    def get_lines(self, a):
        lns = np.array([])
        if isinstance(a, str):
            return [a]
        for l in a:
            lns = np.append(lns, l)
        return lns.tolist()

    def find_nearest(self, claim_doc):
        claim, evidence = claim_doc
        doc = evidence[2]
        tag = evidence[3]
        lines = self.database.get_doc_lines(doc)
        non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if
                           len(line.split("\t")) > 1 and len(line.split("\t")[1].strip())]
        if tag == -2:
            tfidf = OnlineTfidfDocRanker(self.args, [line for line in non_empty_lines], None)
            line_ids, scores = tfidf.closest_docs(claim, 5)
            return doc, np.array(non_empty_lines)[line_ids], line_ids
        else:
            return doc, non_empty_lines[SimpleRandom.get_instance().next_rand(0, len(non_empty_lines) - 1)], []

    def find_nearest_lines(self, claim, pp):
        claims = [claim for i in range(len(pp))]
        with ThreadPool(4) as threads:
            results = threads.map(self.find_nearest, zip(claims, pp))
        return results

    def generate_page_predictions(self, split, k):
        print("Saving prepared dataset to {}".format("{0}_pipeline.ns.pages.p{1}.jsonl".format(split, k)))
        if not os.path.exists(self.out_dir):
            print("Creating directory {}".format(self.out_dir))
            os.makedirs(self.out_dir)
        with open(self.dataset_root + "data/fever-data/{0}.jsonl".format(split), "r") as f_in:
            with open(self.out_dir + "{0}_pipeline.ns.pages.p{1}.jsonl".format(split, k), "w+") as f_out:
                for line in tqdm(f_in.readlines()):
                    line = json.loads(line)

                    doc_names, doc_scores = self.ranker.closest_docs(line['claim'], k)
                    pp = list(doc_names)

                    for idx, evidence_group in enumerate(line['evidence']):
                        for evidence in evidence_group:
                            if idx < len(pp):
                                evidence[2] = pp[idx]
                                # if it belongs to NEI class, set the tag to -1 to indicate a random sentence selection during sentence sampling
                                if line["label"] == "NOT ENOUGH INFO":
                                    evidence[3] = -1
                                else:
                                    evidence[3] = -2
                            else:
                                evidence[2] = pp[-1]  # repeat the last one
                                evidence[3] = -2
                    if len(pp) > idx:
                        for i in range(len(pp) - 1 - idx):
                            ev = [[-1, None, pp[i], -2]]
                            evidence_group.extend(ev)
                    f_out.write(json.dumps(line) + "\n")

    def generate_sentence_predictions(self, split, k):
        print("Saving prepared dataset to {}".format("{0}_pipeline.ps.pages.p{1}.jsonl".format(split, k)))
        with open(self.out_dir + "{0}_pipeline.ns.pages.p{1}.jsonl".format(split, k), "r") as f_in:
            with open(self.out_dir + "{0}_pipeline.ps.pages.p{1}.jsonl".format(split, k), "w+") as f_out:
                for line in tqdm(f_in.readlines()):
                    line = json.loads(line)
                    claim = line['claim']
                    for idx, evidence_group in enumerate(line['evidence']):
                        claims = [claim for i in range(len(evidence_group))]
                        with ThreadPool(4) as threads:
                            results = threads.map(self.find_nearest, zip(claims, evidence_group))

                        line_matches = [r[1] for r in results]

                        line_matches = [self.get_lines(ln) for ln in line_matches]
                        line_ids = [r[2] for r in results]

                        predicted_lines = [[a, b] for a, b in zip(line_matches, line_ids)]
                        ## match the number of lines matches to evidence
                        lines_needed = len(evidence_group)
                        for i in range(len(predicted_lines), lines_needed):
                            predicted_lines[i] = []

                        for idx, evidence in enumerate(evidence_group):
                            evidence.append(predicted_lines[idx])
                    f_out.write(json.dumps(line) + "\n")
