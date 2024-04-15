CREATE KEYSPACE IF NOT EXISTS wikidata WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE wikidata;

CREATE TABLE IF NOT EXISTS articles (
    wiki text,
    language text,
    title text,
    chunk_no int,
    colbert_no int,
    id int,
    revision int,
    body text,
    all_minilm_l6_v2 vector<float, 384>,
    e5_mistral_7b_instruct vector<float, 4096>,
    colbert vector<float, 128>,
    PRIMARY KEY ((wiki, language, title), chunk_no, colbert_no)
) WITH COMMENT = 'chunk_no=-1 contans the full (unchunked) article body. colbert_no=-1 contains the body chunk and all_minilm_l6_v2 and e5_mistral_7b_instruct embeddings.';

CREATE CUSTOM INDEX IF NOT EXISTS all_minilm_l6_v2_ann ON articles(all_minilm_l6_v2) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'COSINE' };

CREATE CUSTOM INDEX IF NOT EXISTS e5_mistral_7b_instruct_ann ON articles(e5_mistral_7b_instruct) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'COSINE' };

# on astra add to OPTIONS 'source_model': 'bert'
CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann ON articles(bert_embedding) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'DOT_PRODUCT' };

