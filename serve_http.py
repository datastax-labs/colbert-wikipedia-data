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

from flask import Flask, request, render_template_string
from serve import retrieve_minilm, retrieve_colbert

app = Flask(__name__)

# Updated HTML template for displaying results as sections with titles and bodies
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <!-- Bootstrap CSS (using a public CDN) -->
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
  <title>Retrieve Models Interface</title>
</head>
<body>
<div class="container">
  <h2>DPR vs ColBERT</h2>
  <form method="post">
    <div class="form-group">
      <label for="queryInput">Enter your query:</label>
      <input type="text" class="form-control" id="queryInput" name="query" required>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
  </form>
  {% if results %}
  <div class="row">
    <div class="col-md-6">
      <h3>DPR (all-MiniLM-L6-v2) Results</h3>
      {% for result in results.ada %}
        <h5>{{ result.title }}</h5>
        <p>{{ result.body }}</p>
      {% endfor %}
    </div>
    <div class="col-md-6">
      <h3>ColBERT Results</h3>
      {% for result in results.colbert %}
        <h5>{{ result.title }}</h5>
        <p>{{ result.body }}</p>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query = request.form['query']
        ada_results = retrieve_minilm(query)  # Ensure this returns a list of dicts with 'title' and 'body'
        colbert_results = retrieve_colbert(query)  # Ensure this returns a list of dicts with 'title' and 'body'
        results = {'ada': ada_results, 'colbert': colbert_results}
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
