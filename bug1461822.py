# Databricks notebook source
from pyspark.sql.functions import pandas_udf, PandasUDFType, countDistinct, col, mean, stddev, expr
from pyspark.sql.types import *
import pandas as pd

from collections import defaultdict, Counter
from urlparse import urlparse
import Levenshtein
import difflib
import json
import random
import re
import requests
from PIL import Image
from StringIO import StringIO

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load data from S3
# MAGIC 
# MAGIC TODO: I should move this into a utility file

# COMMAND ----------

import jsbeautifier
import boto3
import gzip
import io
from botocore.exceptions import ClientError

S3_ROOT = 's3a://openwpm-crawls'

class S3Dataset(object):
  def __init__(self, crawl_directory):
    self._s3_table_loc = "%s/%s/visits/%%s/" % (S3_ROOT, crawl_directory)
    self._s3_content_loc = "%s/%s/content/%%s.gz" % (S3_ROOT, crawl_directory)

    
  def read_table(self, table_name, columns=None):
    table = sqlContext.read.parquet(self._s3_table_loc % table_name)
    if columns is not None:
      return table.select(columns)
    return table
  
  def read_content(self, content_hash):
    """Read the content corresponding to `content_hash`.
    
    NOTE: This can only be run in the driver process since it uses the spark context
    """
    return sc.textFile(self._s3_content_loc % content_hash)
  
  def collect_content(self, content_hash, beautify=False):
    """Collect content for `content_hash` to driver
    
    NOTE: This can only be run in the driver process since it uses the spark context
    """
    content = ''.join(self.read_content(content_hash).collect())
    if beautify:
      return jsbeautifier.beautify(content)
    return content
  
  
class WorkerSafeS3Dataset(object):
  """This class is a helper to allow worker processes to access the S3 dataset.
  
  Workers can not use the spark context directly. This class should include no
  references to the spark context so it can be serialized and sent to workers.
  """
  def __init__(self, crawl_directory):
    self._bucket = S3_ROOT.split('//')[1]
    self._key = "%s/content/%%s.gz" % crawl_directory
    
  def collect_content(self, content_hash, beautify=False):
    """Collect content in worker process.
    
    See: https://github.com/databricks/spark-deep-learning/issues/67#issuecomment-340089028
         for a description of why it's faster to use this than to loop through `read_content`
         in the driver process and then distribute those handles to the worker processes.
    """
    s3 = boto3.client('s3')
    try:
      obj = s3.get_object(
        Bucket=self._bucket,
        Key=self._key % content_hash
      )
      body = obj["Body"]
      compressed_content = io.BytesIO(body.read())
      body.close()
    except ClientError as e:
      if e.response['Error']['Code'] != 'NoSuchKey':
        raise
      else:
        return None
    
    with gzip.GzipFile(fileobj=compressed_content, mode='r') as f:
      content = f.read()
    
    if content is None or content == "":
      return ""
    
    if beautify:
      try:
        content = jsbeautifier.beautify(content)
      except IndexError:
        pass
    try:
      return content.decode('utf-8')
    except ValueError:
      return content.decode('utf-8', errors='ignore')

# COMMAND ----------

from __future__ import absolute_import
from __future__ import print_function
from publicsuffix import PublicSuffixList, fetch
from ipaddress import ip_address
from six.moves.urllib.parse import urlparse
from functools import wraps
import tempfile
import codecs
import os
import six
from six.moves import range

# We cache the Public Suffix List in temp directory
PSL_CACHE_LOC = os.path.join(tempfile.gettempdir(), 'public_suffix_list.dat')


def get_psl(location=PSL_CACHE_LOC):
    """
    Grabs an updated public suffix list.
    """
    if not os.path.isfile(location):
        psl_file = fetch()
        with codecs.open(location, 'w', encoding='utf8') as f:
            f.write(psl_file.read())
    psl_cache = codecs.open(location, encoding='utf8')
    return PublicSuffixList(psl_cache)


def load_psl(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if 'psl' not in kwargs:
            if wrapper.psl is None:
                wrapper.psl = get_psl()
            return function(*args, psl=wrapper.psl, **kwargs)
        else:
            return function(*args, **kwargs)
    wrapper.psl = None
    return wrapper


def is_ip_address(hostname):
    """
    Check if the given string is a valid IP address
    """
    try:
        ip_address(six.text_type(hostname))
        return True
    except ValueError:
        return False


@load_psl
def get_ps_plus_1(url, **kwargs):
    """
    Returns the PS+1 of the url. This will also return
    an IP address if the hostname of the url is a valid
    IP address.
    An (optional) PublicSuffixList object can be passed with keyword arg 'psl',
    otherwise a version cached in the system temp directory is used.
    """
    if 'psl' not in kwargs:
        raise ValueError(
            "A PublicSuffixList must be passed as a keyword argument.")
    hostname = urlparse(url).hostname
    if is_ip_address(hostname):
        return hostname
    elif hostname is None:
        # Possible reasons hostname is None, `url` is:
        # * malformed
        # * a relative url
        # * a `javascript:` or `data:` url
        # * many others
        return
    else:
        return kwargs['psl'].get_public_suffix(hostname)

# COMMAND ----------

DB_BLOCKING_1 = '2018-11-26_bug1461822-blocking-1'
DB_BLOCKING_2 = '2018-11-26_bug1461822-blocking-2'
DB_NON_BLOCKING_1 = '2018-11-26_bug1461822-non-blocking-1'
DB_NON_BLOCKING_2 = '2018-11-26_bug1461822-non-blocking-2'
RESP_COLS = ['visit_id', 'top_level_url', 'url', 'content_hash', 'original_cookies']
REQ_COLS = ['visit_id', 'top_level_url', 'url', 'original_cookies', 'content_policy_type', 'channel_id']
REDIR_COLS = ['visit_id', 'old_channel_id', 'new_channel_id']

block_1 = S3Dataset(DB_BLOCKING_1)
block_2 = S3Dataset(DB_BLOCKING_2)
non_block_1 = S3Dataset(DB_NON_BLOCKING_1)
non_block_2 = S3Dataset(DB_NON_BLOCKING_2)

# COMMAND ----------

block_1_w = WorkerSafeS3Dataset(DB_BLOCKING_1)
block_2_w = WorkerSafeS3Dataset(DB_BLOCKING_2)
non_block_1_w = WorkerSafeS3Dataset(DB_NON_BLOCKING_1)
non_block_2_w = WorkerSafeS3Dataset(DB_NON_BLOCKING_2)

# COMMAND ----------

resp_block_1 = block_1.read_table('http_responses', RESP_COLS)
resp_block_2 = block_2.read_table('http_responses', RESP_COLS)
resp_non_block_1 = non_block_1.read_table('http_responses', RESP_COLS)
resp_non_block_2 = non_block_2.read_table('http_responses', RESP_COLS)

req_block_1 = block_1.read_table('http_requests', REQ_COLS)
req_block_2 = block_2.read_table('http_requests', REQ_COLS)
req_non_block_1 = non_block_1.read_table('http_requests', REQ_COLS)
req_non_block_2 = non_block_2.read_table('http_requests', REQ_COLS)

redir_block_1 = block_1.read_table('http_redirects', REDIR_COLS)
redir_block_2 = block_2.read_table('http_redirects', REDIR_COLS)
redir_non_block_1 = non_block_1.read_table('http_redirects', REDIR_COLS)
redir_non_block_2 = non_block_2.read_table('http_redirects', REDIR_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basic data quality checks

# COMMAND ----------

from pyspark.sql.functions import countDistinct, col, isnan, lit, sum, count
import matplotlib.pyplot as plt

def count_not_null(c, nan_as_null=False):
    """Use conversion between boolean and integer
    - False -> 0
    - True ->  1
    """
    pred = col(c).isNotNull() & (~isnan(c) if nan_as_null else lit(True))
    return sum(pred.cast("integer")).alias(c)

def plot_hist(df):
  bins, counts = df.groupby('visit_id').count().rdd.values().histogram(20)
  fig, ax = plt.subplots()
  plt.hist(bins[:-1], bins=bins, weights=counts)
  display(fig)
  
def check_df(df):
  print(
    "Number of sites %d" %
    df.agg(countDistinct(col("visit_id"))).collect()[0]
  )
  print(
    "Number of records with stripped cookies %d" %
    df.agg(count_not_null('original_cookies')).collect()[0]
  )
  if 'content_hash' in df.columns:
    print(
      "Total number of scripts loaded %d" %
      df.agg(count_not_null("content_hash")).collect()[0]
    )
    print(
      "Number of distinct scripts loaded %d" %
      df.agg(countDistinct(col("content_hash"))).collect()[0]
    )

# COMMAND ----------

plot_hist(resp_block_1)

# COMMAND ----------

plot_hist(resp_block_2)

# COMMAND ----------

plot_hist(resp_non_block_1)

# COMMAND ----------

plot_hist(resp_non_block_2)

# COMMAND ----------

print("\n---Blocking 1")
print("responses")
check_df(resp_block_1)
print("requests")
check_df(req_block_1)

print("\n----Blocking 2")
print("responses")
check_df(resp_block_2)
print("requests")
check_df(req_block_2)

print("\n---Non Blocking 1")
print("responses")
check_df(resp_non_block_1)
print("requests")
check_df(req_non_block_1)

print("\n---Non Blocking 2")
print("responses")
check_df(resp_non_block_2)
print("requests")
check_df(req_non_block_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prep data for comparison

# COMMAND ----------

from urlparse import urlparse
def strip_url(url, scheme=False):
  """Returns a url stripped to (scheme)?+hostname+path"""
  purl = urlparse(url)
  surl = ''
  if scheme:
      surl += purl.scheme + '://'
  try:
      surl += purl.hostname + purl.path
  except TypeError:
      surl += purl.hostname
  return surl
udf_strip_url = udf(strip_url, StringType())
udf_get_ps_plus_1 = udf(get_ps_plus_1, StringType())

def add_redirect_context(df, redir_df):
  """Add context to requests or responses table from redirect table.
  Necessary columns:
    redir table: visit_id, old_channel_id, new_channel_id
    req/resp table: visit_id, channel_id
  """
  parent_view = redir_df.select(
    col('visit_id'),
    col('old_channel_id').alias('parent_channel_id'),
    col('new_channel_id').alias('channel_id')
  )
  child_view = redir_df.select(
    col('visit_id'),
    col('old_channel_id').alias('channel_id'),
    col('new_channel_id').alias('child_channel_id')
  )
  join_cols = ['visit_id', 'channel_id']
  out = df.join(parent_view, join_cols, how="left")
  out = out.join(child_view, join_cols, how="left")
  return out

# COMMAND ----------

# Add analysis columns to response tables
b1 = resp_block_1.alias('b1')
b2 = resp_block_2.alias('b2')
n1 = resp_non_block_1.alias('n1')
n2 = resp_non_block_2.alias('n2')
b1 = b1.withColumn("surl", udf_strip_url("url"))
b2 = b2.withColumn("surl", udf_strip_url("url"))
n1 = n1.withColumn("surl", udf_strip_url("url"))
n2 = n2.withColumn("surl", udf_strip_url("url"))
b1 = b1.withColumn("ps1", udf_get_ps_plus_1("url"))
b2 = b2.withColumn("ps1", udf_get_ps_plus_1("url"))
n1 = n1.withColumn("ps1", udf_get_ps_plus_1("url"))
n2 = n2.withColumn("ps1", udf_get_ps_plus_1("url"))

# COMMAND ----------

# Add analysis columns to request tables
req_b1 = req_block_1.alias('b1')
req_b2 = req_block_2.alias('b2')
req_n1 = req_non_block_1.alias('n1')
req_n2 = req_non_block_2.alias('n2')
req_b1 = req_b1.withColumn("surl", udf_strip_url("url"))
req_b2 = req_b2.withColumn("surl", udf_strip_url("url"))
req_n1 = req_n1.withColumn("surl", udf_strip_url("url"))
req_n2 = req_n2.withColumn("surl", udf_strip_url("url"))
req_b1 = req_b1.withColumn("ps1", udf_get_ps_plus_1("url"))
req_b2 = req_b2.withColumn("ps1", udf_get_ps_plus_1("url"))
req_n1 = req_n1.withColumn("ps1", udf_get_ps_plus_1("url"))
req_n2 = req_n2.withColumn("ps1", udf_get_ps_plus_1("url"))
req_b1 = add_redirect_context(req_b1, redir_block_1)
req_b2 = add_redirect_context(req_b2, redir_block_2)
req_n1 = add_redirect_context(req_n1, redir_non_block_1)
req_n2 = add_redirect_context(req_n2, redir_non_block_2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Comparison of resources loaded

# COMMAND ----------

all_top = set(req_b1.select(col('top_level_url')).distinct().collect())
all_top = all_top.intersection(req_b2.select(col('top_level_url')).distinct().collect())
all_top = all_top.intersection(req_n1.select(col('top_level_url')).distinct().collect())
all_top = all_top.intersection(req_n2.select(col('top_level_url')).distinct().collect())
all_top = set([x.top_level_url for x in all_top])

# COMMAND ----------

def get_average_num_ps1s(df):
  return df.where(
    col('top_level_url').isin(all_top)
  ).groupBy('top_level_url').agg(countDistinct('ps1').alias("count")).select(
    mean('count').alias('mean'),
    expr('percentile_approx(count, 0.5)').alias('median'),
    stddev('count').alias('std')
  ).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we check the mean, median, and standard deviation in the number of distinct eTLD+1s loaded on each top-level domain. As you can see, the crawls that block cookies (labeled `b1` and `b2`) load resources from ~3-4 fewer eTLD+1s than the crawls which do not block cookies (`n1` and `n2`), and that there is agreement between crawls with the same crawl settings.
# MAGIC 
# MAGIC This suggests that blocking tracking cookies does indeed lead to changes in the resources loaded by a site, but we need to drill down into those differences to determine whether they will cause breakage that a user will experience. It's possible this is simply caused by cookie syncing tags not firing (as there are no cookie values to sync).

# COMMAND ----------

print(get_average_num_ps1s(req_b1))

# COMMAND ----------

print(get_average_num_ps1s(req_b2))

# COMMAND ----------

print(get_average_num_ps1s(req_n1))

# COMMAND ----------

print(get_average_num_ps1s(req_n2))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Annotations add details about the context of the load to the URL string. The rest of the comparisons should be generic to the set of included annotations.

# COMMAND ----------

grp_key = [
  'top_level_url'
]
grp_schema = req_b1.select(grp_key).schema
grp_schema.add(StructField("resources", ArrayType(StringType())))
grp_schema.add(StructField("ps1s", ArrayType(StringType())))

@pandas_udf(grp_schema, PandasUDFType.GROUPED_MAP)
def get_resources(pdf):
  return pd.DataFrame.from_records(
    [[
      pdf.top_level_url.iloc[0],
      pdf.surl.unique(),
      [get_ps_plus_1('http://'+x) for x in pdf.surl.unique()]
    ]],
    columns=[
      'top_level_url',
      'resources',
      'ps1s'],
  )
                     
@pandas_udf(grp_schema, PandasUDFType.GROUPED_MAP)
def get_annotated_resources(pdf):
  resource_set = set()
  ps1_set = set()
  for rowid, row in pdf.iterrows():
    # Prepend context to the url string
    context = "%s:%s:%%s" % (
      row['content_policy_type'],
      row['parent_channel_id'] is not None and row['parent_channel_id'] != ''
    )
    resource_set.add(context % row['surl'])
    ps1_set.add(context % get_ps_plus_1(row['url']))
  return pd.DataFrame.from_records(
    [[
      pdf.top_level_url.iloc[0],
      list(resource_set),
      list(ps1_set)
    ]],
    columns=[
      'top_level_url',
      'resources',
      'ps1s'],
  )

# COMMAND ----------

# Run this cell for analysis without annotations
rb1 = req_b1.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_resources).alias('rb1')
rb2 = req_b2.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_resources).alias('rb2')
rn1 = req_n1.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_resources).alias('rn1')
rn2 = req_n2.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_resources).alias('rn2')

# COMMAND ----------

# Run this cell for analysis with annotations
rb1 = req_b1.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_annotated_resources).alias('rb1')
rb2 = req_b2.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_annotated_resources).alias('rb2')
rn1 = req_n1.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_annotated_resources).alias('rn1')
rn2 = req_n2.where(
  col('top_level_url').isin(all_top)
).groupby(grp_key).apply(get_annotated_resources).alias('rn2')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can estimate the set of missing resources by (on a per-site basis) looking for URLs that are loaded in both crawls that don't block cookies but not in either of the crawls that block cookies. These differences can also be caused by differences in load time or simply random chance, but acts as a good initial filter.
# MAGIC 
# MAGIC If this is run with annotated URLS -- that include whether the load was the result of a redirect as well as the content type of the load -- we can make some inference on how high of breakage to expect. If the resources that failed to load are images, there is less of a breakage concern than if they are Scripts or CSS content. We can also spot check the top-level URLs to determine whether any user-visible breakage is observed when content blocking is enabled.

# COMMAND ----------

def compute_missing(b1_items, b2_items, n1_items, n2_items):
  return list(set(n1_items).intersection(set(n2_items)).difference(set(b1_items).union(set(b2_items))))
udf_compute_missing = udf(compute_missing, ArrayType(StringType()))

# COMMAND ----------

JOIN_COLUMNS = ['top_level_url']
SELECT_COLUMNS = JOIN_COLUMNS + ['resources', 'ps1s']
resources = (
  rb1[SELECT_COLUMNS]
  .join(rb2[SELECT_COLUMNS], JOIN_COLUMNS)
  .join(rn1[SELECT_COLUMNS], JOIN_COLUMNS)
  .join(rn2[SELECT_COLUMNS], JOIN_COLUMNS)
)

# COMMAND ----------

resources = resources.select(
  col('top_level_url'),
  col('rb1.resources').alias('b1_resources'),
  col('rb1.ps1s').alias('b1_ps1s'),
  col('rb2.resources').alias('b2_resources'),
  col('rb2.ps1s').alias('b2_ps1s'),
  col('rn1.resources').alias('n1_resources'),
  col('rn1.ps1s').alias('n1_ps1s'),
  col('rn2.resources').alias('n2_resources'),
  col('rn2.ps1s').alias('n2_ps1s'),
)

# COMMAND ----------

resources = resources.withColumn(
  'missing_resources',
  udf_compute_missing('b1_resources', 'b2_resources', 'n1_resources', 'n2_resources')
)
resources = resources.withColumn(
  'missing_ps1s',
  udf_compute_missing('b1_ps1s', 'b2_ps1s', 'n1_ps1s', 'n2_ps1s')
)

# COMMAND ----------

missing_resources = resources[
  resources.missing_resources.isNotNull() |
  resources.missing_ps1s.isNotNull()
][['top_level_url', 'missing_resources', 'missing_ps1s']].toPandas()

# COMMAND ----------

missed_urls = Counter()
missed_urls_set = defaultdict(set)
missed_ps1s = Counter()
missed_ps1s_set = defaultdict(set)
for rowid, row in missing_resources.iterrows():
  for item in row['missing_resources']:
    missed_urls[item] += 1
    missed_urls_set[item].add(row['top_level_url'])
  for item in row['missing_ps1s']:
    missed_ps1s[item] += 1
    missed_ps1s_set[item].add(row['top_level_url'])

# COMMAND ----------

# Non annotated results
missed_urls.most_common(100)

# COMMAND ----------

# Non annotated results
missed_ps1s.most_common(100)

# COMMAND ----------

# Annotated results
missed_urls_df = pd.DataFrame.from_records(
  [x[0].split(':', 2) + [x[1]] for x in missed_urls_set.items()],
  columns=["content_policy_type", "from_redirect", "url", "fps"]
)
missed_urls_df['count'] = missed_urls_df.fps.apply(len)
missed_ps1s_df = pd.DataFrame.from_records(
  [x[0].split(':', 2) + [x[1]] for x in missed_ps1s_set.items()],
  columns=["content_policy_type", "from_redirect", "ps1", "fps"]
)
missed_ps1s_df['count'] = missed_ps1s_df.fps.apply(len)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Missing images
# MAGIC 
# MAGIC A `content_policy_type` of 3 is an image. We can examine the types of missing images by re-requesting missing image content and examining the resulting image size. Since we've stripped query string from URLs and are re-requesting the images, we expect a bunhc of the lookups to fail. We could lower this failure set by sampling in full URLs with query strings for each URL that errored until we have a successful request (or run out of URLs). The full URL would serve as a representative example of the others.

# COMMAND ----------

def get_image_size(url):
  if not url.startswith('http'):
    url = 'http://'+url
  try:
    resp = requests.get(url)
  except Exception:
    return
  if resp.status_code != 200:
    return
  try:
    img = Image.open(StringIO(resp.content))
  except IOError:
    return
  return img.size

# COMMAND ----------

sizes = dict()
errored_urls = set()
count = 0
for url in missed_urls_df[missed_urls_df.content_policy_type == '3'].sort_values('count', ascending=False)['url']:
  count += 1
  if count % 50 == 0:
    print("Processed %d images" % count)
  size = get_image_size(url)
  if size is None:
    errored_urls.add(url)
    continue
  sizes[url] = size

# COMMAND ----------

print("Number of successfully requested URLs: %d\nNumber of lookup failures: %d" % (len(sizes), len(errored_urls)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We see that the most commonly missed URLs are 1x1 pixels, which are likely used for cookie syncing / tracking.

# COMMAND ----------

size_counts = Counter()
for k, v in sizes.items():
  size_counts[v] += 1

# COMMAND ----------

size_counts.most_common(20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC If we weight the count by the number of sites on which each missed resource was loaded, we see that pixels are the overwhelming majority of missed resources between the two crawls. We don't expect missing pixels to lead to breakage.
# MAGIC 
# MAGIC 
# MAGIC As we go down the list we see many common ad formats:
# MAGIC 
# MAGIC     250 x 250 – Square
# MAGIC     200 x 200 – Small Square
# MAGIC     468 x 60 – Banner
# MAGIC     728 x 90 – Leaderboard
# MAGIC     300 x 250 – Inline Rectangle
# MAGIC     336 x 280 – Large Rectangle
# MAGIC     120 x 600 – Skyscraper
# MAGIC     160 x 600 – Wide Skyscraper
# MAGIC     300 x 600 – Half-Page Ad
# MAGIC     970 x 90 – Large Leaderboard
# MAGIC Note that we can't interpret this as ads which were missed due to the cookie blocking. It may simply be the case that a different ad image was served in the blocked cookie crawls than in the non blocked crawls.

# COMMAND ----------

missed_urls_df['size'] = missed_urls_df.url.apply(lambda x: sizes.get(x, None))

# COMMAND ----------

instance_count = Counter()
for rowid, row in missed_urls_df[missed_urls_df.content_policy_type == '3'].iterrows():
  if row['size'] is None:
    instance_count['UNKNOWN'] += row['count']
    continue
  instance_count[row['size']] += row['count']

# COMMAND ----------

instance_count.most_common(20)

# COMMAND ----------

# Annotated results
missed_ps1s.most_common(100)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Missing (non-image) content
# MAGIC 
# MAGIC Mapping from integer to type: https://searchfox.org/mozilla-central/source/dom/base/nsIContentPolicy.idl

# COMMAND ----------

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)  

# COMMAND ----------

missed_urls_df.groupby('content_policy_type').url.count().sort_values(ascending=False)

# COMMAND ----------

missed_urls_df.groupby('content_policy_type')['count'].sum().sort_values(ascending=False)

# COMMAND ----------

for item in missed_urls_df.content_policy_type.unique():
  total = missed_urls_df[missed_urls_df.content_policy_type == item]['count'].sum()
  ads = missed_urls_df[
    (missed_urls_df.content_policy_type == item) & (
      missed_urls_df.url.str.contains('ad', case=False) | 
      missed_urls_df.url.str.contains('pixel', case=False) | 
      missed_urls_df.url.str.contains('cookie', case=False) | 
      missed_urls_df.url.str.contains('banner', case=False) | 
      missed_urls_df.url.str.contains('px', case=False) | 
      missed_urls_df.url.str.contains('uid', case=False) |
      missed_urls_df.url.str.contains('sync', case=False) |
      missed_urls_df.url.str.contains('match', case=False) |
      missed_urls_df.url.str.contains('tag', case=False) |
      missed_urls_df.url.str.contains('beacon', case=False) |
      missed_urls_df.url.str.contains(r'\d{1,3}[xX]\d{1,3}')
    )
  ]['count'].sum()
  print("%-6s - %0.2f (%d/%d) instances contain ad-related words" % (item, float(ads)/total, ads, total))

# COMMAND ----------

not_ads = missed_urls_df[
  ~(missed_urls_df.url.str.contains('ad', case=False) |
    missed_urls_df.url.str.contains('pixel', case=False) | 
    missed_urls_df.url.str.contains('cookie', case=False) | 
    missed_urls_df.url.str.contains('banner', case=False) | 
    missed_urls_df.url.str.contains('px', case=False) | 
    missed_urls_df.url.str.contains('uid', case=False) |
    missed_urls_df.url.str.contains('sync', case=False) |
    missed_urls_df.url.str.contains('match', case=False) |
    missed_urls_df.url.str.contains('tag', case=False) |
    missed_urls_df.url.str.contains('beacon', case=False) |
    missed_urls_df.url.str.contains(r'\d{1,3}[xX]\d{1,3}')
   )
]

# COMMAND ----------

# Identity providers
not_ads[(
    not_ads.url.str.contains('facebook.com') | 
    not_ads.url.str.contains('facebook.net') |
    not_ads.url.str.contains('google.com') | 
    not_ads.url.str.contains('twitter.com') | 
    not_ads.url.str.contains('disqus.com')
  )][['url', 'count']].sort_values('count', ascending=False)[0:20]

# COMMAND ----------

# Identity providers
not_ads[(
    not_ads.url.str.contains('like') | 
    not_ads.url.str.contains('login') | 
    not_ads.url.str.contains('connect') | 
    not_ads.url.str.contains('oauth')
  )][['url', 'count']].sort_values('count', ascending=False)[0:20]

# COMMAND ----------

# Sites to manually check
def print_fps(tp_url):
  fps = missed_urls_df[missed_urls_df.url == tp_url].fps.iloc[0]
  print("%s\n%s" % (tp_url, random.sample(fps, min(len(fps), 5))))
  
print_fps('www.google.com/recaptcha/api2/bframe')
print_fps('www.facebook.com/fr/b.php')
print_fps('accounts.google.com/o/oauth2/postmessageRelay')
print_fps('www.facebook.com/plugins/like.php')
print_fps('connect.facebook.net/en_US/all.js')  # loads when checked

# COMMAND ----------

# Subdocuments
not_ads[(not_ads.content_policy_type == '7')].sample(50)['url']

# COMMAND ----------

# Scripts
not_ads[(not_ads.content_policy_type == '2')].sample(50)['url']

# COMMAND ----------

# XHR
not_ads[not_ads.content_policy_type == '11'].sample(50)['url']

# COMMAND ----------

# Stylesheet
not_ads[not_ads.content_policy_type == '4'].sample(50)['url']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Direct hash comparisons

# COMMAND ----------

JOIN_COLUMNS = ['top_level_url', 'surl']
SELECT_COLUMNS = JOIN_COLUMNS + ['content_hash']
responses = (
  b1[b1.content_hash.isNotNull()][SELECT_COLUMNS]
  .join(b2[b2.content_hash.isNotNull()][SELECT_COLUMNS], JOIN_COLUMNS)
  .join(n1[n1.content_hash.isNotNull()][SELECT_COLUMNS], JOIN_COLUMNS)
  .join(n2[n2.content_hash.isNotNull()][SELECT_COLUMNS], JOIN_COLUMNS)
)

# COMMAND ----------

# Rename the content hash columns to include a prefix
responses = responses.select(
  col('top_level_url'),
  col('surl'),
  col('b1.content_hash').alias('b1_content_hash'),
  col('b2.content_hash').alias('b2_content_hash'),
  col('n1.content_hash').alias('n1_content_hash'),
  col('n2.content_hash').alias('n2_content_hash')
)

# COMMAND ----------

presence = responses.groupby('surl').agg(countDistinct('top_level_url').alias("count")).sort(col("count").desc()).collect()

# COMMAND ----------

results = responses.where(
  (col('b1_content_hash') == col('b2_content_hash')) &
  (col('n1_content_hash') == col('n2_content_hash')) & 
  (col('b1_content_hash') != col('n1_content_hash'))
).groupby('surl').agg(countDistinct('top_level_url').alias("count")).sort(col("count").desc()).collect()

# COMMAND ----------

temp = Counter(dict(results))

# COMMAND ----------

temp.most_common(100)

# COMMAND ----------

with open('/dbfs/FileStore/senglehardt/top_scripts.csv', 'w') as f:
  for url, count in dict(results).items():
    f.write('%s,%d\n' % (url, count))

# COMMAND ----------

def get_diff(content_1, content_2):
  if content_1 is None or content_2 is None:
    return
  content_1 = content_1.split('\n')
  content_2 = content_2.split('\n')
  return u'\n'.join(difflib.context_diff(content_1, content_2))

grp_key = [
  #'top_level_url',
  'surl'
]
grp_schema = responses.select(grp_key).schema
#grp_schema.add(StructField("diffs", ArrayType(StringType())))
grp_schema.add(StructField("diffs", StringType()))
@pandas_udf(grp_schema, PandasUDFType.GROUPED_MAP)
def get_diffs(pdf):
  diffs = set()
  for rowid, row in pdf.iterrows():
    b1_content = block_1_w.collect_content(row['b1_content_hash'], True)
    n1_content = non_block_1_w.collect_content(row['n1_content_hash'], True)
    if b1_content is None or n1_content is None:
      continue
    diff = get_diff(b1_content, n1_content)
    if diff is None:
      continue 
    diffs.add(diff)
  return pd.DataFrame.from_records(
    [[
      #pdf.top_level_url.iloc[0],
      pdf.surl.iloc[0],
      '|X|X|'.join(diffs)
    ]],
    columns=[
      #'top_level_url',
      'surl',
      'diffs'],
  )

# COMMAND ----------

results = responses.where(
  (col('b1_content_hash') == col('b2_content_hash')) &
  (col('n1_content_hash') == col('n2_content_hash')) & 
  (col('b1_content_hash') != col('n1_content_hash'))
).groupby(grp_key).apply(get_diffs).write.csv('dbfs:/FileStore/senglehardt/bug1461822-diffs-csv/')

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/senglehardt/bug1461822-diffs-csv/'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Edit distance threshold comparisons
# MAGIC 
# MAGIC I'm using the Levenshtein package in a UDF to calculate this ratio, but it should be possible to do it natively in spark (which is likely much faster).
# MAGIC 
# MAGIC Spark provides a `levenshtein` function that computes the edit distance. The ratio is computed from that as `(lensum - levenshtein)/lensum` where `lensum` is the sum of the lengths of the two strings. Thus, we could chain the builtin functions like:
# MAGIC 
# MAGIC ```(F.length(F.concat_ws('', 'b1_content', 'n1_content')) - F.levenshtein('b1_content', 'n1_content')) / F.length(F.concat_ws('', 'b1_content', 'n1_content'))```
# MAGIC 
# MAGIC **This analysis suffers from the [small files problem](http://garrens.com/blog/2017/11/04/big-data-spark-and-its-small-files-problem/). I'm going to shelf this until we solve: https://github.com/mozilla/OpenWPM/issues/247.**

# COMMAND ----------

def compute_similarity(content_1, content_2):
  if content_1 is None or content_2 is None:
    return 0
  return Levenshtein.ratio(content_1, content_2)
udf_compute_similarity = udf(compute_similarity, FloatType())

# COMMAND ----------

responses = responses.withColumn("b1_content", udf(block_1_w.collect_content, StringType())('b1_content_hash')).cache()
responses = responses.withColumn("b2_content", udf(block_2_w.collect_content, StringType())('b2_content_hash')).cache()
responses = responses.withColumn("n1_content", udf(non_block_1_w.collect_content, StringType())('n1_content_hash')).cache()
responses = responses.withColumn("n2_content", udf(non_block_2_w.collect_content, StringType())('n2_content_hash')).cache()
responses = responses.withColumn("blocked_similarity", udf_compute_similarity('b1_content', 'b2_content'))
responses = responses.withColumn("non_blocked_similarity", udf_compute_similarity('n1_content', 'n2_content'))
responses = responses.withColumn("cross_similarity", udf_compute_similarity('b1_content', 'n1_content'))
responses = responses.drop("b1_content")
responses = responses.drop("b2_content")
responses = responses.drop("n1_content")
responses = responses.drop("n2_content")

# COMMAND ----------

test2 = test.iloc[0:10].copy()

# COMMAND ----------

test2['b1_content'] = test2.b1_content_hash.apply(lambda x: block_1_w.collect_content(x, True))
test2['b2_content'] = test2.b2_content_hash.apply(lambda x: block_2_w.collect_content(x, True))
test2['n1_content'] = test2.n1_content_hash.apply(lambda x: non_block_1_w.collect_content(x, True))
test2['n2_content'] = test2.n2_content_hash.apply(lambda x: non_block_2_w.collect_content(x, True))

# COMMAND ----------

test2['blocked_similarity'] = test2.apply(axis=1, func=lambda x: compute_similarity(x['b1_content'], x['b2_content']))
test2['non_blocked_similarity'] = test2.apply(axis=1, func=lambda x: compute_similarity(x['n1_content'], x['n2_content']))
test2['cross_similarity'] = test2.apply(axis=1, func=lambda x: compute_similarity(x['b1_content'], x['n1_content']))

# COMMAND ----------

print test2[['blocked_similarity', 'non_blocked_similarity', 'cross_similarity']]

# COMMAND ----------

results = responses.where(
  (col('blocked_similarity') > 0.9) &
  (col('non_blocked_similarity') > 0.9) & 
  (col('cross_similarity') < 0.9)
).toPandas()

# COMMAND ----------

results = responses.where(
  (col('blocked_similarity') > 0.9) &
  (col('non_blocked_similarity') > 0.9) & 
  (col('cross_similarity') < 0.9)
).groupby('surl').agg(countDistinct('top_level_url').alias("count")).sort(col("count").desc()).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Playground

# COMMAND ----------

# Define group keys
grp_key = ['surl']
grp_schema = responses.select(grp_key).schema
grp_schema.add(StructField("is_match", BooleanType()))
grp_schema.add(StructField("blocked_hash", StringType()))
grp_schema.add(StructField("non_blocked_hash", StringType()))
@pandas_udf(grp_schema, PandasUDFType.GROUPED_MAP)
def get_flagged_scripts(pdf):
  # Check if scripts are the same for blocked, different for not blocked
  b1_hashes = pdf['b1_content_hash'].unique()
  b2_hashes = pdf['b2_content_hash'].unique()
  n1_hashes = pdf['n1_content_hash'].unique()
  n2_hashes = pdf['n2_content_hash'].unique()
  
  is_match = b1_hashes == b2_hashes and n1_hashes == n2_hashes and b1_hashes != b2_hashes
  
  # Grab groupby columns
  surl = pdf.surl.iloc[0]
  
  # Return a dataframe with results (pandas_udf requires a DataFrame return type)
  return pd.DataFrame(
    {
      'surl': surl,
      'is_match': is_match,
      'blocked_hash': b1_hashes.pop(),
      'non_blocked_hash': n1_hashes.pop()
    },
    columns=[
      'surl', 'is_match', 'blocked_hash', 'non_blocked_hash'
    ],
    index=[0]
  )

# COMMAND ----------

responses.where(
  (col('b1.content_hash') == col('b2.content_hash')) &
  (col('n1.content_hash') == col('n2.content_hash')) & 
  (col('b1.content_hash') != col('n1.content_hash'))
)[['top_level_url','surl', 'b1.content_hash', 'n1.content_hash']].show(100, False)

# COMMAND ----------

def print_diff(b1_hash, n1_hash):
  c1 = jsbeautifier.beautify(block_1.collect_content(b1_hash)).split('\n')
  c2 = jsbeautifier.beautify(non_block_1.collect_content(n1_hash)).split('\n')
  for line in difflib.context_diff(c1, c2):
    print line
    
def get_distance(b1_hash, n1_hash):
  c1 = block_1.collect_content(b1_hash, True)
  c2 = jsbeautifier.beautify(non_block_1.collect_content(n1_hash))
  return Levenshtein.ratio(c1, c2)

# COMMAND ----------

block_1.read_content('fe19d34621060b736ce2ccef5a768483')

# COMMAND ----------

print_diff('fe19d34621060b736ce2ccef5a768483', '2cc1ae51707c3218ee60fdc48f77f31e')

# COMMAND ----------

print_distance('fe19d34621060b736ce2ccef5a768483', '2cc1ae51707c3218ee60fdc48f77f31e')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `sb.scorecardresearch.com/beacon.js` that appears to change the way it loads an external tag depending on cookie availability.

# COMMAND ----------

print_diff('d5fde2b15cdc97016b5eec04cd23dd47', 'c4968e93227e4e9e7ac8f4e2a3c83c76')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `tredir.go.com/capmon/GetDE/` sends back a different location string based on the cookie value

# COMMAND ----------

print_diff('994e5e150a7bdda18057a102f740c72a', 'c52478e04cc1c84b32e23769dba878d3')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `apis.google.com/js/plusone.js` renders a different plus 1 button?

# COMMAND ----------

print_diff('3a10074541476de6e7ad160c014811f5', '77c39204857e3d94b0140c9d0aac07d0')

# COMMAND ----------

test = get_diff('3a10074541476de6e7ad160c014811f5', '77c39204857e3d94b0140c9d0aac07d0')

# COMMAND ----------

print get_diff('asdfasdfasdfa', 'asdfasfddsfads')

# COMMAND ----------

'sdfasdfasdfasdfsadf'.split('\n')

# COMMAND ----------

