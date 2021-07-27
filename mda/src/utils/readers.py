from tqdm import tqdm
import json

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
