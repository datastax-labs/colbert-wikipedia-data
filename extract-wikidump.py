#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 DataStax
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# from a Wikipedia Cirrus dump
#  chunk articles,
#   creates all-MiniLM-L6-v2 and colbert embeddings,
#    and writes results in Apache Cassandra.  See schema.cql
#


from db import DB

import argparse
import gzip
import itertools
import json
import logging
import os.path
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


thread_local_storage = threading.local()

# e5-mistral-7b-instruct is very slow and resource intensive
MISTRAL_ENABLED = True


# langchain sentence chunking
def _chunk_string(s):
    """Divide a string into chunks of `chunk_length` with overlaps of `chunk_overlap`."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    return [chunk.page_content for chunk in splitter.create_documents([s])]


def create_transformers():
    _get_threadlocal_transformer_minilm()
    _get_threadlocal_encoder_colbert()
    if (MISTRAL_ENABLED):
        _get_threadlocal_transformer_mistral()
    print("transformers created")

def _get_threadlocal_transformer_minilm():
    """ https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 """
    if getattr(thread_local_storage, "transformer_minilm", None) is None:
        thread_local_storage.transformer_minilm = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return thread_local_storage.transformer_minilm


def _get_threadlocal_transformer_mistral():
    """ https://huggingface.co/wvprevue/e5-mistral-7b-instruct """
    if getattr(thread_local_storage, "transformer_mistral", None) is None:
        thread_local_storage.transformer_mistral = SentenceTransformer(
            "wvprevue/e5-mistral-7b-instruct"
        )
    return thread_local_storage.transformer_mistral


def _get_threadlocal_encoder_colbert():
    """ """
    if getattr(thread_local_storage, "encoder_colbert", None) is None:
        cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
        cp = Checkpoint(cf.checkpoint, colbert_config=cf)
        thread_local_storage.encoder_colbert = CollectionEncoder(cf, cp)
    return thread_local_storage.encoder_colbert


def process_dump(input):
    # download transformers first in the main thread. prevents parallel downloads wastage
    create_transformers()

    num_threads = 1
    counter = itertools.count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while True:
            # input format
            # {"index":{"_type":"page","_id":"3825914"}}
            # {"namespace":0,"title":TITLE,"timestamp":"2014-06-29T15:51:09Z","text":TEXT,...}
            line = input.readline()
            if not line: break
            index = json.loads(line)
            content = json.loads(input.readline())
            executor.submit(_process_article, index, content, counter)


def _process_article(index, content, counter):
    type = index["index"]["_type"]
    if type == "_doc" and content["namespace"] == 0:
        id = int(index["index"]["_id"])
        language = content["language"]
        wiki = content["wiki"]
        revision = int(content["version"])
        title = content["title"]
        body = re.sub(r"  \^ .*", "", content["text"]).replace("'", "")
        c = f"{title}\n\n{body}"
        
        #db.session.execute(f"""
        #    INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, body)
        #    VALUES ('{wiki}', '{language}', '{title}', -1, -1, {id}, {revision}, '{c}')
        #    """)

        # chunk
        chunks = _chunk_string(body)
        # create embeddings
        minilm_embeddings = _get_threadlocal_transformer_minilm().encode(chunks, show_progress_bar=False)
        mistral_embeddings = _get_threadlocal_transformer_mistral().encode(chunks, show_progress_bar=False) if MISTRAL_ENABLED else [0]

        print(f"chunks {len(chunks)} minilm {len(minilm_embeddings)} mistral {len(mistral_embeddings)}")
        # write each chunk to separate file
        for chunk_no, chunk in enumerate(chunks):
            minilm_e = minilm_embeddings[chunk_no].tolist()
            mistral_e = mistral_embeddings[chunk_no].tolist() if MISTRAL_ENABLED else None
            c = f"{title}\n\n{chunk}"

            if MISTRAL_ENABLED:
                #db.session.execute(
                print(
                    f"""
                    INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, body, all_minilm_l6_v2, e5_mistral_7b_instruct)
                    VALUES ('{wiki}', '{language}', '{title}', {chunk_no}, -1, {id}, {revision}, '{c}', {minilm_e}, {mistral_e})
                    """)
            else:
                #db.session.execute(
                print(
                    f"""
                    INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, body, all_minilm_l6_v2)
                    VALUES ('{wiki}', '{language}', '{title}', {chunk_no}, -1, {id}, {revision}, '{c}', {minilm_e})
                    """)

        # colbert. this is noisy, xxx how to quiet it ?
        encoder_colbert = _get_threadlocal_encoder_colbert()
        embeddings_flat, counts = encoder_colbert.encode_passages(chunks)

        # split up embeddings_flat by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(counts[:-1]))
        embeddings_by_part = [embeddings_flat[start:start+count] for start, count in zip(start_indices, counts)]
        for chunk_no, embeddings in enumerate(embeddings_by_part):
            for bert_embedding_no, e in enumerate(embeddings):
                e = e.tolist()
                # chunk text is not stored under the colbert embeddings, to save on storage
                #  use `bert_embedding_no = -1` to get chunk text
                #db.session.execute(
                #    f"""
                #    INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, colbert)
                #    VALUES ('{wiki}', '{language}', '{title}', {chunk_no}, {bert_embedding_no}, {id}, {revision}, {e} )
                #    """)

    processed = next(counter)
    if processed % 1000 == 0:
        print(f"PROCESSED {processed}")

def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "input", help="Cirrus json wiki dump file (or '-' for reading from stdin)"
    )
    groupS = parser.add_argument_group("Special")
    groupS.add_argument(
        "-q", "--quiet", action="store_true", help="suppress reporting progress info"
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s")

    if not args.quiet:
        logging.getLogger().setLevel(logging.INFO)

    input_file = args.input
    
    if input_file == "-":
        input = sys.stdin
    elif input_file.endswith(".gz"):
        input = gzip.open(input_file)
    else:
        input = open(input_file)

    process_dump(input)


db = DB(protocol_version=5)

if __name__ == "__main__":
    main()
