CREATE KEYSPACE IF NOT EXISTS wikidata WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE wikidata;

CREATE TABLE IF NOT EXISTS articles (
    wiki text,
    language text,
    title text,
    chunk_no int,
    bert_embedding_no int,
    id int,
    revision int,
    body text,
    all_minilm_l6_v2_embedding vector<float, 384>,
    bert_embedding vector<float, 128>,
    PRIMARY KEY ((wiki, language, title), chunk_no, bert_embedding_no)
) WITH COMMENT = 'chunk_no=-1 contans the full (unchunked) article body. bert_embedding_no=-1 contains the body chunk and the all_minilm_l6_v2_embeddings.';

CREATE CUSTOM INDEX IF NOT EXISTS all_minilm_l6_v2_ann ON articles(all_minilm_l6_v2_embedding) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'COSINE' };

CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann ON articles(bert_embedding) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'DOT_PRODUCT' };

