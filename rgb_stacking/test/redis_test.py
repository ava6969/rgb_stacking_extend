
from redis.cluster import RedisCluster as Redis
r = Redis(host='localhost', port=6379)
print(r.get_nodes())
request = r.set('foo', 'bar')
response = r.get('foo')
print(response)
