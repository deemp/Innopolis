import psycopg2

con = psycopg2.connect(database="customers", user="postgres", password="password", host="localhost", port="5432")

print("Database opened successfully")

cur = con.cursor()

import re
def print_cost(cur):
    analyze_fetched = str(cur.fetchall())
    cost = re.findall('cost=\d+.\d+..\d+.\d+', analyze_fetched)[0]
    print(cost)

#1. some query
cur.execute('''Explain ANALYZE SELECT * FROM customer WHERE LEFT(name,2) = 'AB';
''')
print_cost(cur)

cur.execute('''Explain ANALYZE SELECT * FROM customer_not_indexed WHERE LEFT(name,2) = 'AB';
''')
print_cost(cur)

cur.execute('''Explain ANALYZE SELECT * FROM customer WHERE 10 < age and age < 20;
''')
print_cost(cur)

cur.execute('''Explain ANALYZE SELECT * FROM customer_not_indexed WHERE 10 < age and age < 20;
''')
print_cost(cur)
print(str(cur.fetchall()))

#no difference

#2.
cur.execute('''Explain ANALYZE SELECT * FROM customer_not_indexed WHERE 10 < age and age < 20;
''')
print_cost(cur)
