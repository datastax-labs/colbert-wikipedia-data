# RAG Apps using ColBERT DPR on wikipedia datasets

ColBERT Dense Passage Retrieval (DPR) with ready-to-go pre-vectorised wikipedia datasets for RAG applications.

Datasets contain both ColBERT and all-MiniLM-L6-v2 embeddings, on 1024 character chunks for all articles in the wikipedia set.  Datasets from different wikipedia sites and languages can be combined and searched in the one table and index.

These datasets are intended for
 - RAG Apps using either/or ColBERT and all-MiniLM-L6-v2 embeddings for DPR on wikipedia data,
 - comparing and benchmarking ColBERT vs all-MiniLM-L6-v2 performance and relevancy, 
 - production use.


ColBERT demonstrates improved performanced and accuracy for RAG applications and uses a smaller model. Despite the need to create more embeddings: also resulting in a larger ANN vector index; and having to multiple more searches per request, the improved relevancy and lower system resource cost of ColBERT makes it the attractive solution.

The datasets can be found here:
 https://drive.google.com/drive/folders/1GTr5R2E3y97ZXaSG7t-XM45MqgjkSI3s?usp=share_link


# Setup #

* Python setup

```
cd colbert-wikipedia-data
virtualenv -p python3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* Database setup


You can skip this step if you already have Apache Cassandra >=5.0-beta2 running.
```
# 5.0-beta2 not yet released.  use latest 5.0 nightlies build for now.
#wget https://www.apache.org/dyn/closer.lua/cassandra/5.0-beta2/apache-cassandra-5.0-beta2-bin.tar.gz

wget https://nightlies.apache.org/cassandra/cassandra-5.0/Cassandra-5.0-artifacts/214/Cassandra-5.0-artifacts/jdk=jdk_11_latest,label=cassandra/build/apache-cassandra-5.0-beta2-SNAPSHOT-bin.tar.gz

tar -xzf  apache-cassandra-5.0*-bin.tar.gz
rm apache-cassandra-5.0*-bin.tar.gz
cp apache-cassandra-5.0*/conf/cassandra_latest.yaml apache-cassandra-5.0*/conf/cassandra.yaml
export PATH="$(echo $(pwd)/apache-cassandra-5.0*)/bin/:$PATH"
export CASSANDRA_DATA="$(echo $(pwd)/apache-cassandra-5.0*)/data"
cassandra -f
```

All following steps assume C* is listening on localhost:9042

 


* Load schema and the prepared dataset for the simple-english wikipedia dataset
```
cqlsh -f schema.cql

# Download (from a browser) https://drive.google.com/file/d/1CcMMsj8jTKRVGep4A7hmOSvaPepsaKYP/view?usp=share_link
# these files are very big, tens/hundreds of GBs

# move the downloaded file to the current directory, renaming it to simplewiki-sstable.tar
# for example:
mv ~/Downloads/simplewiki-20240304-sstable.tar simplewiki-sstable.tar

# note, if you have existing data in this table you'll want to check the tarball's files don't clobber any existing
tar -xf simplewiki-sstable.tar -C ${CASSANDRA_DATA}/data/wikidata/articles-*/

# alternative is to just restart the node (any failures in the indexes will be rebuilt automatically)
nodetool import wikidata articles ${CASSANDRA_DATA}/data/wikidata/articles-*/
```


The datamodel is a single table `wikidata.articles`.
Separate ANN SAI indexes exist for the minilm-l6-v2 and colbert embeddings.

# Serve ColBERT and all-MiniLM-L6-v2 DPR


* Download the ColBERT model from https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz and extract
  it to the checkpoints/ subdirectory

```
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
mkdir checkpoints
tar -xvf colbertv2.0.tar.gz -C checkpoints/
```

* Run command line
```
python serve.py
```

* Run webserver
```
python serve_httpy.py`

# open http://localhost:5000
```


The Retriver is simple code off this datamodel, fetching similarity search results from both ColBERT and all-MiniLM-L6-v2 embeddings.

# FAQ

# I only want the ColBERT embeddings

Just drop the `all_minilm_l6_v2_ann` index and then the `all_minilm_l6_v2_embedding` column.

```
cqlsh

DROP INDEX wikidata.all_minilm_l6_v2_ann ;
ALTER TABLE wikidata.articles DROP all_minilm_l6_v2_embedding ;
```

If you only want the all_minilm_l6_v2 embeddings then it is the same procedure but for the `colbert_ann` index and `bert_embedding` column.  Note this will leave all the bert_embedding_no rows behind but they will be empty.


# Manual extraction of wikipedia datasets

If you want to extract the wikipedia data yourself (instead of downloading the above ready prepared sstable data)

```
cqlsh -e 'DROP INDEX wikidata.all_minilm_l6_v2_ann ; DROP INDEX wikidata.colbert_ann ;'
nodetool disableautocompaction

python extract-wikidump.py -q simplewiki-20240304-cirrussearch-content.json

nodetool compact
# to watch progress (ctl-c when complete)
watch nodetool compactionstats

cqlsh -e "CREATE CUSTOM INDEX all_minilm_l6_v2_ann ON articles(all_minilm_l6_v2_embedding) USING 'StorageAttachedIndex' WITH OPTIONS = { 'similarity_function': 'COSINE' };"
cqlsh -e "CREATE CUSTOM INDEX colbert_ann ON articles(bert_embedding) USING 'StorageAttachedIndex' WITH OPTIONS = { 'similarity_function': 'DOT_PRODUCT' };"

# to watch progress (ctl-c when complete)
watch nodetool compactionstats
```


Extraction (extract-wikidump.py) works with wikipedia cirrus dumps found at https://dumps.wikimedia.org/other/cirrussearch/

The extraction defaults to chunks 1024, with overlap 256, characters using langchain's RecursiveCharacterTextSplitter.
