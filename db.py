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

from cassandra.cluster import Cluster

keyspace = "wikidata"

class DB:
    def __init__(self, **kwargs):
        self.cluster = Cluster(**kwargs)
        self.session = self.cluster.connect()
        self.session.default_timeout = 60

        query_minilm_cql = f"""
        SELECT title, chunk_no, body
        FROM {keyspace}.articles
        ORDER BY all_minilm_l6_v2_embedding ANN OF ?
        LIMIT 5
        """
        self.query_minilm_stmt = self.session.prepare(query_minilm_cql)

        query_colbert_ann_cql = f"""
        SELECT title, chunk_no, bert_embedding, body
        FROM {keyspace}.articles
        ORDER BY bert_embedding ANN OF ?
        LIMIT 5
        """
        self.query_colbert_ann_stmt = self.session.prepare(query_colbert_ann_cql)

        query_part_by_pk = f"""
        SELECT body
        FROM {keyspace}.articles
        WHERE wiki = 'simplewiki' and language = 'en' and title = ? AND chunk_no = ? AND bert_embedding_no = -1
        """
        self.query_part_by_pk_stmt = self.session.prepare(query_part_by_pk)
