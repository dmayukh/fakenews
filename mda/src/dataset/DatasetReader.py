import json
from tqdm import tqdm
import unicodedata
import re
import numpy as np
import tensorflow as tf
import pickle
from sklearn import preprocessing
from drqa.retriever import DocDB, utils

class LabelSchema:

    def __init__(self,labels):
        self.labels = {self.preprocess(val):idx for idx,val in enumerate(labels)}
        self.idx = {idx:self.preprocess(val) for idx,val in enumerate(labels)}

    def get_id(self,label):
        if self.preprocess(label) in self.labels:
            return self.labels[self.preprocess(label)]
        return None

    def preprocess(self,item):
        return item.lower()

class FEVERLabelSchema(LabelSchema):

    def __init__(self):
        super().__init__(["supports", "refutes", "not enough info"])

class Reader:

    def __init__(self,encoding="utf-8"):
        self.enc = encoding

    def read(self,file):
        with open(file,"r",encoding = self.enc) as f:
            return self.process(f)

    def process(self,f):
        pass

class JSONLineReader(Reader):

    def process(self,fp):
        data = []
        for line in tqdm(fp.readlines()):
            data.append(json.loads(line.strip()))
        return data

class train_line_formatter():

    def format(self, lines):
        formatted = []
        for line in tqdm(lines):
            fl = self.format_line(line)
            if fl is not None:
                if isinstance(fl,list):
                    formatted.extend(fl)
                else:
                    formatted.append(fl)
        return formatted

    def format_line(self, line):
        label_schema = FEVERLabelSchema()
        # get the label, i.e. SUPPORTS etc.
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]
        pages = []
        # did we get the closest sentences to the claim text? is this the sentence or the line number from the doc text?
        if 'predicted_sentences' in line:
            pages.extend([(ev[0], ev[1]) for ev in line["predicted_sentences"]])
        elif 'predicted_pages' in line:
            pages.extend([(ev[0], -1) for ev in line["predicted_pages"]])
        else:
            # these are the human annotated evidence available in the original training file
            for evidence_group in line["evidence"]:
                pages.extend([(ev[2], ev[3]) for ev in evidence_group])
        return {"claim": line["claim"], "evidence": pages, "label": label_schema.get_id(annotation),
                "label_text": annotation}

class test_line_formatter():

    def format(self, lines):
        formatted = []
        for line in tqdm(lines):
            fl = self.format_line(line)
            if fl is not None:
                if isinstance(fl,list):
                    formatted.extend(fl)
                else:
                    formatted.append(fl)
        return formatted

    def format_line(self, line):
        label_schema = FEVERLabelSchema()
        # get the label, i.e. SUPPORTS etc.
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]
        pages = []
        lines = []
        # did we get the closest sentences to the claim text? is this the sentence or the line number from the doc text?
        if 'predicted_sentences' in line:
            pages.extend([(ev[0], ev[1]) for ev in line["predicted_sentences"]])
        elif 'predicted_pages' in line:
            pages.extend([(ev[0], -1) for ev in line["predicted_pages"]])
        else:
            # only if evidence[0] is > -1, we have relevant predicted lines in evidence[4]
            for evidence_group in line["evidence"]:
                pages.extend([(ev[2], ev[3]) for ev in evidence_group])
            for evidence_group in line["evidence"]:
                for ev in evidence_group:
                    if ev[0] > -1:
                        lines.extend(ev[4][0])
        return {"claim": line["claim"], "evidence": pages, "lines": lines, "label": label_schema.get_id(annotation),
                "label_text": annotation}


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '[START] ' + w + ' [END]'
    return w


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


class DatasetReader:

    #split = 'paper_dev', working_dir = 'working/data/', k = 5
    #working_dir + "training/{0}_pipeline.ps.pages.p{1}.jsonl".format(split, k)
    def __init__(self, in_file, label_checkpoint_file, database_path, type='train'):
        self.lineformatter = None
        self.type = type
        if self.type == 'train':
            self.lineformatter = train_line_formatter()
        else:
            self.lineformatter = test_line_formatter()
        self.formatter = self.lineformatter
        self.reader = JSONLineReader()
        self.in_file = in_file
        self.data = None
        self.label_checkpoint_file = label_checkpoint_file
        # database_path = '/local/fever-common/data/fever/fever.db'
        self.database_path = database_path
        self.database = FeverDocDB(self.database_path)
        self.labelencoder = None

    def read(self):
        raw = self.reader.read(self.in_file)
        self.data = self.lineformatter.format(raw)
        return raw, self.data

    def read_labels(self):
        #checkpoint_file = 'working/data/training/label_encoder_train.pkl'
        labels = [d['label_text'] for d in self.data]
        if self.label_checkpoint_file != None:
            with open(self.label_checkpoint_file, 'rb') as f:
                self.labelencoder = pickle.load(f)
        else:
            labels = [d['label_text'] for d in self.data]
            self.labelencoder = preprocessing.LabelEncoder()
            self.labelencoder.fit(labels)
        labels_enc = self.labelencoder.transform(labels)
        test_labels = np.zeros(shape=(len(labels_enc), 3))
        for idx, val in enumerate(labels_enc):
            test_labels[idx][val] = 1
        return self.labels_to_tensors(test_labels)

    def labels_to_tensors(self, labels):
        lbls = tf.reshape(tf.convert_to_tensor(labels, dtype=tf.int32), (labels.shape))
        lbls_ds = tf.data.Dataset.from_tensor_slices(lbls)
        return lbls_ds

    def get_dataset(self):
        if self.type == 'train':
            ds = self.get_train_dataset()
        else:
            ds = self.get_test_dataset()
        labels = self.read_labels()
        return tf.data.Dataset.zip((ds, labels))

    def get_train_data_generator(self):
        for data in self.data:
            claim = preprocess(data["claim"])
            body_ids = [e[0] for e in data["evidence"]]
            bodies = [self.database.get_doc_text(id) for id in set(body_ids)]
            parts = [claim, " ".join(bodies)]
            yield claim, " ".join(parts)

    def get_test_data_generator(self):
        for d in self.data:
            claim = preprocess(d["claim"])
            lines = d["lines"]
            yield claim, " ".join(lines)

    def get_train_dataset(self):
        generator = lambda: self.get_train_data_generator()
        return tf.data.Dataset.from_generator(
            generator, output_signature=(
                tf.TensorSpec(shape=(2,), dtype=tf.string)))

    def get_test_dataset(self):
        generator = lambda: self.get_test_data_generator()
        return tf.data.Dataset.from_generator(
            generator, output_signature=(
                tf.TensorSpec(shape=(2,), dtype=tf.string)))
