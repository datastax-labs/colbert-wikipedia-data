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

        #
        # Queries
        #

        query_minilm_cql = f"""
        SELECT title, chunk_no, body
        FROM {keyspace}.articles
        ORDER BY all_minilm_l6_v2 ANN OF ?
        LIMIT 5
        """
        try:
            self.query_minilm_stmt = self.session.prepare(query_minilm_cql)
        except:
            print("could not prepare ANN on all_minilm_l6_v2 (mising index ?)")

        query_colbert_ann_cql = f"""
        SELECT title, chunk_no, colbert
        FROM {keyspace}.articles
        ORDER BY colbert ANN OF ?
        LIMIT 5
        """
        try:
            self.query_colbert_ann_stmt = self.session.prepare(query_colbert_ann_cql)
        except:
            print("could not prepare ANN on colbert (mising index ?)")

        query_part_by_pk = f"""
        SELECT body
        FROM {keyspace}.articles
        WHERE wiki = 'simplewiki' and language = 'en' and title = ? AND chunk_no = ? AND colbert_no = -1
        """
        self.query_part_by_pk_stmt = self.session.prepare(query_part_by_pk)

        #
        # Inserts
        #

        insert_article_cql = f"""
            INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, body)
            VALUES ('?', '?', '?', -1, -1, ?, ?, '?')
            """
        self.insert_article_stmt = self.session.prepare(insert_article_cql)

        insert_chunk_cql = f"""
                INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, body, all_minilm_l6_v2, e5_mistral_7b_instruct)
                VALUES ('?', '?', '?', ?, -1, ?, ?, '?', ?, ?)
                """
        self.insert_chunk_stmt = self.session.prepare(insert_chunk_cql)

        insert_colbert_cql = f"""
                    INSERT INTO wikidata.articles (wiki, language, title, chunk_no, colbert_no, id, revision, colbert)
                    VALUES ('?', '?', '?', ?, ?, ?, ?, ?)
                    """
        self.insert_article_stmt = self.session.prepare(insert_colbert_cql)
