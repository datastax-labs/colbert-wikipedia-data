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

import time
from db import DB
from torch import tensor
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from sentence_transformers import SentenceTransformer


db = DB(protocol_version=5)

_cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
_cp = Checkpoint(_cf.checkpoint, colbert_config=_cf)
encode = lambda q: _cp.queryFromText([q])[0]


transformer_minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def minilm_embedding_of(text: str):
    return transformer_minilm.encode(text)

def retrieve_minilm(query):
    qv = minilm_embedding_of(query)
    rows = db.session.execute(db.query_minilm_stmt, [qv])
    return [{'title': row.title, 'chunk_no': row.chunk_no, 'body': row.body} for row in rows]


def maxsim(qv, embeddings):
    return max(qv @ e for e in embeddings)

def retrieve_colbert(query):
    query_encodings = encode(query)
    # find the most relevant documents
    embeddings_for_part = {}
    bodies = {}
    scores = {}
    for qv in query_encodings:
        rows = db.session.execute(db.query_colbert_ann_stmt, [list(qv)])
        for row in rows:
            title = row.title
            chunk_no = str(row.chunk_no)
            t = tuple([title, chunk_no])
            if not t in embeddings_for_part:
                embeddings_for_part[t] = []
                bodies[t] = db.session.execute_async(db.query_part_by_pk_stmt, [title, int(chunk_no)])
            embeddings_for_part[t].append(tensor(row.bert_embedding))
    # score each document
    for title, chunk_no in embeddings_for_part.keys():
        t = tuple([title, chunk_no])
        scores[t] = sum(maxsim(qv, embeddings_for_part[t]) for qv in query_encodings)

    # load the source chunk for the top 5
    docs_by_score = sorted(scores, key=scores.get, reverse=True)[:5]
    L = []
    for title, chunk_no in docs_by_score:
        t = tuple([title, chunk_no])
        rows = bodies[t].result()
        L.append({'title': title, 'chunk_no': chunk_no, 'body': rows.one().body})
    return L


def format_stdout(L):
    return '\n\n'.join(f"{i+1}. {row['title']} [chunk {row['chunk_no']}]\n{row['body']}" for i, row in enumerate(L))

if __name__ == '__main__':
    while True:
        query = input('Enter a query: ')
        t = time.time()
        results = retrieve_colbert(query)
        t = time.time() - t
        print(f'\n# Retrieving from ColBERT (took {t})#\n')
        print(format_stdout(results))

        t = time.time()
        results = retrieve_minilm(query)
        t = time.time() - t
        print(f'\n\n# Retrieving from all-MiniLM-L6-v2 (took {t})#\n')
        print(format_stdout(results))

        print()
