# A USEFUL SCRIPT USED TO UNDERSTAND HOW GENSIM WORK IN OLD VERSIONS


#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import logging


#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processes

#pretrained word embeddings
pretrained_emb = "toydata/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

#input corpus
train_corpus = "toydata/train_docs.txt"

#output model
saved_path = "toydata/model.bin"

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
print(docs)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)

#save model
model.save(saved_path)


#parameters

model="./toydata/doc2vec.bin"
test_corpus="./toydata/test_docs.txt"
output_file="./toydata/test_vectors1.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]
print(test_docs)

count = 0

#infer test vectors
output = open(output_file, "w")
for d in test_docs:
    if count == 0:
        print(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
        count = 1
    output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output.flush()
output.close()
